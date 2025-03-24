import numpy as np
import math
from scipy.optimize import fsolve
import random as rnd


# 1) Basic Classes: Node and Loop


class Node:
    """
    Represents a node in a pipe network.
    """
    def __init__(self, Name='a', Pipes=None, ExtFlow=0):
        self.name = Name
        self.pipes = Pipes if Pipes else []
        self.extFlow = ExtFlow  # L/s internally
        self.QNet = 0
        self.P = 0
        self.oCalculated = False

    def getNetFlowRate(self):
        """Sum flows from connected pipes plus external flow."""
        Qtot = self.extFlow
        for p in self.pipes:
            Qtot += p.getFlowIntoNode(self.name)
        self.QNet = Qtot
        return self.QNet

    def setExtFlow(self, E, SI=True):
        """
        Sets external flow (E). If SI=False, E is in cfs => convert to L/s.
        """
        from_unit = 1.0 if SI else UC.ft3_to_L
        self.extFlow = E * from_unit


class Loop:
    """
    Defines a loop in the pipe network.
    """
    def __init__(self, Name='A', Pipes=None):
        self.name = Name
        self.pipes = Pipes if Pipes else []

    def getLoopHeadLoss(self):
        """
        Compute net head loss around the loop (m of fluid).
        """
        deltaP = 0.0
        if not self.pipes:
            return deltaP
        startNode = self.pipes[0].startNode
        for p in self.pipes:
            phl = p.getFlowHeadLoss(startNode)
            deltaP += phl
            # move to next node in the loop
            startNode = p.endNode if startNode != p.endNode else p.startNode
        return deltaP


# 2) Utility Classes: UC (unit converter) and Fluid


class UC:
    """
    A utility class for handling unit conversions (English <-> SI).
    """
    ft_to_m = 1 / 3.28084
    ft2_to_m2 = ft_to_m ** 2
    ft3_to_m3 = ft_to_m ** 3
    ft3_to_L  = ft3_to_m3 * 1000
    L_to_ft3  = 1 / ft3_to_L
    in_to_m   = ft_to_m / 12
    m_to_in   = 1 / in_to_m
    in2_to_m2 = in_to_m ** 2
    m2_to_in2 = 1 / in2_to_m2

    g_SI      = 9.80665
    g_EN      = 32.174
    gc_EN     = 32.174
    gc_SI     = 1.0
    lbf_to_kg = 1 / 2.20462
    lbf_to_N  = lbf_to_kg * g_SI
    pa_to_psi = (1 / lbf_to_N) * in2_to_m2

    @classmethod
    def viscosityEnglishToSI(cls, mu):
        """
        lb*s/ft^2 -> Pa*s
        """
        cf = (1 / cls.ft2_to_m2) * cls.lbf_to_kg * cls.g_SI
        return mu * cf

    @classmethod
    def densityEnglishToSI(cls, rho):
        """
        lb/ft^3 -> kg/m^3
        """
        cf = cls.lbf_to_kg / cls.ft3_to_m3
        return rho * cf

    @classmethod
    def m_to_psi(cls, h, rho):
        """
        Convert fluid height (m) to psi.
        """
        p_pa = rho * cls.g_SI * h
        return p_pa * cls.pa_to_psi

    @classmethod
    def psi_to_m(cls, p, rho):
        """
        Convert psi to fluid height (m).
        """
        pa = p / cls.pa_to_psi
        return pa / (rho * cls.g_SI)


class Fluid:
    """
    Fluid with dynamic viscosity mu and density rho in SI.
    If constructed with SI=False, we do internal conversions.
    """
    def __init__(self, mu=0.00089, rho=1000, SI=True):
        if SI:
            self.mu = mu
            self.rho = rho
        else:
            self.mu = UC.viscosityEnglishToSI(mu)
            self.rho = UC.densityEnglishToSI(rho)
        self.nu = self.mu / self.rho



# 3) Pipe Class


class Pipe:
    """
    A pipe with orientation from the lower-letter node to higher-letter node.
    """
    def __init__(self, Start='A', End='B', L=100, D=200, r=0.00025, fluid=None, SI=True):
        self.startNode = min(Start.lower(), End.lower())
        self.endNode   = max(Start.lower(), End.lower())

        if fluid is None:
            fluid = Fluid()  # default water
        self.fluid = fluid

        # conversions
        if SI:
            self.length = L
            self.rough  = r
            self.d      = D / 1000.0
        else:
            self.length = L * UC.ft_to_m
            self.rough  = r * UC.ft_to_m
            self.d      = D * UC.in_to_m

        self.relrough = self.rough / self.d
        self.A = math.pi * (self.d / 2.0) ** 2
        self.Q = 10.0  # L/s initial guess
        self.vel = 0.0
        self.reynolds = 0.0
        self.hl = 0.0

    def V(self):
        """
        Velocity in m/s. Q is in L/s => convert to m^3/s => Q/1000.
        """
        q_m3s = self.Q / 1000.0
        if self.A < 1e-12:
            self.vel = 0.0
        else:
            self.vel = q_m3s / self.A
        return self.vel

    def Re(self):
        """
        Reynolds number = rho*|V|*d / mu
        """
        v = abs(self.V())
        if v > 1e-12 and self.d > 1e-12:
            self.reynolds = (self.fluid.rho * v * self.d) / self.fluid.mu
        else:
            self.reynolds = 0.0
        return self.reynolds

    def FrictionFactor(self):
        """
        Darcy-Weisbach friction factor, handling laminar (<2000),
        turbulent (>4000), and linear interpolation in between.
        """
        Re = self.Re()
        rr = self.relrough

        def lam():
            return 64.0 / Re if Re > 1e-12 else 0.0

        def CB():
            def eqn(var):
                fval = var[0]
                return (1.0 / math.sqrt(fval)
                        + 2.0 * math.log10(rr / 3.7 + 2.51 / (Re * math.sqrt(fval))))
            sol = fsolve(eqn, [0.01])
            return sol[0]

        if Re <= 2000:
            return lam()
        elif Re >= 4000:
            return CB()
        else:
            f_lam = lam()
            f_col = CB()
            w = (Re - 2000.0) / (4000.0 - 2000.0)
            return f_lam + w * (f_col - f_lam)

    def frictionHeadLoss(self):
        """
        Darcy-Weisbach: hL = f*(L/d)*(v^2/(2*g)), in m of fluid.
        """
        g = 9.81
        ff = self.FrictionFactor()
        v = self.V()
        self.hl = ff * (self.length / self.d) * (v**2 / (2.0 * g))
        return self.hl

    def getFlowHeadLoss(self, s):
        """
        Signed head loss: depends on loop traversal direction vs. flow direction.
        """
        nTraverse = 1 if s == self.startNode else -1
        nFlow = 1 if self.Q >= 0 else -1
        return nTraverse * nFlow * self.frictionHeadLoss()

    def Name(self):
        return f"{self.startNode}-{self.endNode}"

    def oContainsNode(self, node):
        return node in [self.startNode, self.endNode]

    def printPipeFlowRate(self, SI=True):
        q_val = self.Q if SI else (self.Q * UC.L_to_ft3)
        q_units = 'L/s' if SI else 'cfs'
        print(f"The flow in segment {self.Name()} is {q_val:.2f} ({q_units}) and Re={self.reynolds:.1f}")

    def printPipeHeadLoss(self, SI=True):
        """
        Show pipe length, diameter, and head loss in mm or inches.
        """
        self.frictionHeadLoss()
        if SI:
            Lshow = self.length
            Dshow = self.d * 1000.0
            HL    = self.hl * 1000.0
            print(f"head loss in pipe {self.Name()} (L={Lshow:.2f} m, d={Dshow:.2f} mm) "
                  f"is {HL:.2f} mm of water")
        else:
            Lshow = self.length / UC.ft_to_m
            Dshow = self.d * UC.m_to_in
            HL    = self.hl * UC.m_to_in
            print(f"head loss in pipe {self.Name()} (L={Lshow:.2f} in, d={Dshow:.2f} in) "
                  f"is {HL:.2f} in of water")

    def getFlowIntoNode(self, n):
        """
        +Q if flow enters node n, -Q if flow leaves node n.
        """
        return -self.Q if n == self.startNode else self.Q



# 4) PipeNetwork and Setup


class PipeNetwork:
    """
    A container for all pipes, nodes, loops, and a reference fluid.
    """
    def __init__(self, pipes=None, loops=None, nodes=None, fluid=None):
        self.pipes = pipes if pipes else []
        self.loops = loops if loops else []
        self.nodes = nodes if nodes else []
        self.Fluid = fluid if fluid else Fluid()
        self.refNode = 'h'  # reference node for mass-balance skipping

    def buildNodes(self):
        """Auto-generate node objects from pipe endpoints."""
        for p in self.pipes:
            if not self.nodeBuilt(p.startNode):
                self.nodes.append(Node(p.startNode, self.getNodePipes(p.startNode)))
            if not self.nodeBuilt(p.endNode):
                self.nodes.append(Node(p.endNode, self.getNodePipes(p.endNode)))

    def nodeBuilt(self, nodeName):
        return any(n.name == nodeName for n in self.nodes)

    def getNodePipes(self, nodeName):
        return [p for p in self.pipes if p.oContainsNode(nodeName)]

    def getNode(self, name):
        for nd in self.nodes:
            if nd.name == name:
                return nd
        return None

    def getPipe(self, label):
        for p in self.pipes:
            if p.Name() == label:
                return p
        return None

    def findFlowRates(self):
        """
        Solve system: (#nodes - 1) mass balance eqns + (#loops) loop eqns => # pipes unknowns
        """
        q_init = [p.Q for p in self.pipes]

        def residuals(q):
            # update flows
            for i, p in enumerate(self.pipes):
                p.Q = q[i]
            eqs = []
            # node eqns (skip refNode)
            for nd in self.nodes:
                if nd.name.lower() != self.refNode.lower():
                    eqs.append(nd.getNetFlowRate())
            # loop eqns => sum of head losses = 0
            for lp in self.loops:
                eqs.append(lp.getLoopHeadLoss())
            return eqs

        fsolve(residuals, q_init)

    def getNodePressures(self, knownNodeP, knownNode):
        """
        Assign node pressures by naive loop pass, then shift so knownNode has knownNodeP (m).
        """
        # reset
        for nd in self.nodes:
            nd.P = 0.0
            nd.oCalculated = False

        # traverse each loop
        for lp in self.loops:
            if not lp.pipes:
                continue
            sNode = lp.pipes[0].startNode
            startNodeObj = self.getNode(sNode)
            currentP = startNodeObj.P
            startNodeObj.oCalculated = True

            for p in lp.pipes:
                phl = p.getFlowHeadLoss(sNode)
                currentP -= phl
                sNode = p.endNode if sNode != p.endNode else p.startNode
                self.getNode(sNode).P = currentP

        # shift all
        knownRef = self.getNode(knownNode)
        dP = knownNodeP - knownRef.P
        for nd in self.nodes:
            nd.P += dP

    def printPipeFlowRates(self, SI=True):
        for p in self.pipes:
            p.printPipeFlowRate(SI=SI)

    def printNetNodeFlows(self, SI=True):
        for nd in self.nodes:
            Q = nd.QNet if SI else (nd.QNet * UC.L_to_ft3)
            units = "L/s" if SI else "cfs"
            print(f"net flow into node {nd.name} is {Q:.2f} ({units})")

    def printLoopHeadLoss(self, SI=True):
        for lp in self.loops:
            hl = lp.getLoopHeadLoss()
            if SI:
                print(f"head loss for loop {lp.name} is {hl:.2f} (m of water)")
            else:
                cf = UC.m_to_psi(1.0, self.Fluid.rho)
                print(f"head loss for loop {lp.name} is {hl*cf:.2f} (psi)")

    def printPipeHeadLoss(self, SI=True):
        for p in self.pipes:
            p.printPipeHeadLoss(SI=SI)

    def printNodePressures(self, SI=True):
        for nd in self.nodes:
            if SI:
                print(f"Pressure at node {nd.name} = {nd.P:.2f} m of water")
            else:
                cf = UC.m_to_psi(1.0, self.Fluid.rho)
                print(f"Pressure at node {nd.name} = {nd.P*cf:.2f} psi")


def create_network(SIUnits=False):
    """
    Build and configure the pipe network with 13 pipes, external flows, and loops.
    Returns a PipeNetwork object.
    """
    water = Fluid(mu=2.05e-5, rho=62.3, SI=False)  # English units

    # Roughness values in ft
    r_CI = 0.00085
    r_CN = 0.003

    net = PipeNetwork(fluid=water)

    # Create the 13 pipes
    net.pipes.extend([
        Pipe('a','b', 1000, 18, r_CN, water, SI=SIUnits),
        Pipe('a','h', 1600, 24, r_CN, water, SI=SIUnits),
        Pipe('b','c',  500, 18, r_CN, water, SI=SIUnits),
        Pipe('b','e',  800, 16, r_CI, water, SI=SIUnits),
        Pipe('c','d',  500, 18, r_CN, water, SI=SIUnits),
        Pipe('c','f',  800, 16, r_CI, water, SI=SIUnits),
        Pipe('d','g',  800, 16, r_CI, water, SI=SIUnits),
        Pipe('e','f',  500, 12, r_CI, water, SI=SIUnits),
        Pipe('e','i',  800, 18, r_CN, water, SI=SIUnits),
        Pipe('f','g',  500, 12, r_CI, water, SI=SIUnits),
        Pipe('g','j',  800, 18, r_CN, water, SI=SIUnits),
        Pipe('h','i', 1000, 24, r_CN, water, SI=SIUnits),
        Pipe('i','j', 1000, 24, r_CN, water, SI=SIUnits),
    ])

    # Build node objects
    net.buildNodes()

    # External flows in cfs
    net.getNode('h').setExtFlow(10,  SI=False)
    net.getNode('e').setExtFlow(-3,  SI=False)
    net.getNode('f').setExtFlow(-5,  SI=False)
    net.getNode('d').setExtFlow(-2,  SI=False)

    # Build loops
    net.loops.extend([
        Loop('A', [net.getPipe('a-b'), net.getPipe('b-e'),
                   net.getPipe('e-i'), net.getPipe('h-i'), net.getPipe('a-h')]),
        Loop('B', [net.getPipe('b-c'), net.getPipe('c-f'),
                   net.getPipe('e-f'), net.getPipe('b-e')]),
        Loop('C', [net.getPipe('c-d'), net.getPipe('d-g'),
                   net.getPipe('f-g'), net.getPipe('c-f')]),
        Loop('D', [net.getPipe('e-i'), net.getPipe('i-j'),
                   net.getPipe('g-j'), net.getPipe('f-g'), net.getPipe('e-f')])
    ])

    return net



# 5) Main Routine


def main():
    """
    This reorganized code yields the same final output as before.
    """
    # 1) Create the network
    PN = create_network(SIUnits=False)

    # 2) Solve flows
    PN.findFlowRates()

    # 3) Set node h => 80 psi
    knownP_m = UC.psi_to_m(80, PN.Fluid.rho)
    PN.getNodePressures(knownNode='h', knownNodeP=knownP_m)

    # 4) Print results
    PN.printPipeFlowRates(SI=False)
    print()
    print("Check node flows:")
    PN.printNetNodeFlows(SI=False)
    print()
    print("Check loop head loss:")
    PN.printLoopHeadLoss(SI=False)
    print()
    PN.printPipeHeadLoss(SI=False)
    print()
    PN.printNodePressures(SI=False)


if __name__ == "__main__":
    main()
