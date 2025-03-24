import numpy as np
import math
from scipy.optimize import fsolve
import random as rnd


class UC():  # a units conversion class
    """
    A helper class for handling various unit conversions between
    English (ft, in, lb, psi, cfs) and SI (m, kg, Pa, L/s).
    """
    # Class constants
    ft_to_m = 1 / 3.28084
    ft2_to_m2 = ft_to_m ** 2
    ft3_to_m3 = ft_to_m ** 3
    ft3_to_L = ft3_to_m3 * 1000
    L_to_ft3 = 1 / ft3_to_L
    in_to_m = ft_to_m / 12
    m_to_in = 1 / in_to_m
    in2_to_m2 = in_to_m ** 2
    m2_to_in2 = 1 / in2_to_m2
    g_SI = 9.80665  # m/s^2
    g_EN = 32.174  # ft/s^2
    gc_EN = 32.174  # lbm·ft / (lbf·s^2)
    gc_SI = 1.0  # kg·m / (N·s^2)
    lbf_to_kg = 1 / 2.20462
    lbf_to_N = lbf_to_kg * g_SI
    pa_to_psi = (1 / lbf_to_N) * in2_to_m2

    @classmethod
    def viscosityEnglishToSI(cls, mu, toSI=True):
        cf = (1 / cls.ft2_to_m2) * cls.lbf_to_kg * cls.g_SI
        return mu * cf if toSI else mu / cf

    @classmethod
    def densityEnglishToSI(cls, rho, toSI=True):
        cf = cls.lbf_to_kg / cls.ft3_to_m3
        return rho * cf if toSI else rho / cf

    @classmethod
    def m_to_psi(cls, h, rho):
        pa = rho * cls.g_SI * h  # in Pa
        return pa * cls.pa_to_psi

    @classmethod
    def psi_to_m(cls, p, rho):
        pa = p / cls.pa_to_psi
        h = pa / (rho * cls.g_SI)
        return h


class Fluid():
    def __init__(self, mu=0.00089, rho=1000, SI=True):
        if SI:
            self.mu = mu
            self.rho = rho
        else:
            self.mu = UC.viscosityEnglishToSI(mu)
            self.rho = UC.densityEnglishToSI(rho)
        self.nu = self.mu / self.rho


class Node():
    def __init__(self, Name='a', Pipes=None, ExtFlow=0):
        self.name = Name
        self.pipes = Pipes if Pipes else []
        self.extFlow = ExtFlow  # stored in L/s internally
        self.QNet = 0
        self.P = 0.0  # pressure head at the node in m of fluid
        self.oCalculated = False

    def getNetFlowRate(self):
        Qtot = self.extFlow
        for p in self.pipes:
            Qtot += p.getFlowIntoNode(self.name)
        self.QNet = Qtot
        return self.QNet

    def setExtFlow(self, E, SI=True):
        if SI:
            self.extFlow = E
        else:
            self.extFlow = E * UC.ft3_to_L


class Loop():
    def __init__(self, Name='A', Pipes=None):
        self.name = Name
        self.pipes = Pipes if Pipes else []

    def getLoopHeadLoss(self):
        deltaP = 0.0
        if not self.pipes:
            return 0.0
        startNode = self.pipes[0].startNode
        for p in self.pipes:
            phl = p.getFlowHeadLoss(startNode)
            deltaP += phl
            if startNode == p.startNode:
                startNode = p.endNode
            else:
                startNode = p.startNode
        return deltaP


class Pipe():
    def __init__(self, Start='A', End='B', L=100, D=200, r=0.00025, fluid=None, SI=True):
        # Orientation: lowest letter = startNode.
        self.startNode = min(Start.lower(), End.lower())
        self.endNode = max(Start.lower(), End.lower())
        if SI:
            self.length = L  # L in m
            self.d = D / 1000.0  # D in mm -> m
            self.rough = r  # roughness in m
        else:
            # Now assume L is given in inches (as in the output sample)
            self.length = L * UC.in_to_m  # Convert inches to m
            self.d = D * UC.in_to_m  # D in inches -> m
            self.rough = r * UC.ft_to_m  # roughness is given in ft, convert to m
        self.fluid = fluid if fluid else Fluid()
        self.relrough = self.rough / self.d
        self.A = math.pi * (self.d ** 2) / 4.0
        self.Q = 10.0  # initial guess (internally in L/s)
        self.vel = self.V()
        self.reynolds = self.Re()
        self.hl = 0.0

    def V(self):
        self.vel = (abs(self.Q) / 1000.0) / self.A
        return self.vel

    def Re(self):
        self.reynolds = (self.fluid.rho * self.V() * self.d) / self.fluid.mu
        return self.reynolds

    def FrictionFactor(self):
        Re = self.Re()
        rr = self.relrough

        def colebrook(ff):
            return 1.0 / math.sqrt(ff) + 2.0 * math.log10(rr / 3.7 + 2.51 / (Re * math.sqrt(ff)))

        def solve_colebrook():
            result = fsolve(colebrook, 0.01)
            return result[0]

        def laminar():
            return 64.0 / Re

        if Re <= 2000:
            return laminar()
        elif Re >= 4000:
            return solve_colebrook()
        else:
            f_lam = laminar()
            f_turb = solve_colebrook()
            frac = (Re - 2000.0) / (4000.0 - 2000.0)
            mean = f_lam + (f_turb - f_lam) * frac
            sig = 0.2 * mean * (1 - abs(frac - 0.5) * 2)
            return rnd.normalvariate(mean, sig)

    def frictionHeadLoss(self):
        g = 9.81
        ff = self.FrictionFactor()
        v2 = (self.V()) ** 2
        self.hl = ff * (self.length / self.d) * (v2 / (2.0 * g))
        return self.hl

    def getFlowHeadLoss(self, s):
        nTraverse = 1 if s == self.startNode else -1
        nFlow = 1 if self.Q >= 0 else -1
        return nTraverse * nFlow * self.frictionHeadLoss()

    def Name(self):
        return self.startNode + '-' + self.endNode

    def oContainsNode(self, node):
        return (node == self.startNode) or (node == self.endNode)

    def printPipeFlowRate(self, SI=True):
        if SI:
            q_val = self.Q
            q_units = 'L/s'
        else:
            q_val = self.Q * UC.L_to_ft3
            q_units = 'cfs'
        print("The flow in segment {} is {:0.2f} ({}) and Re={:.1f}".format(
            self.Name(), q_val, q_units, self.reynolds))

    def printPipeHeadLoss(self, SI=True):
        self.frictionHeadLoss()
        if SI:
            L_val = self.length
            d_val = self.d * 1000.0
            hl_val = self.hl * 1000.0
            print("head loss in pipe {} (L={:.2f} m, d={:.2f} mm) is {:.2f} mm of water".format(
                self.Name(), L_val, d_val, hl_val))
        else:
            L_val = self.length * UC.m_to_in
            d_val = self.d * UC.m_to_in
            hl_val = self.hl * UC.m_to_in
            print("head loss in pipe {} (L={:.2f} in, d={:.2f} in) is {:.2f} in of water".format(
                self.Name(), L_val, d_val, hl_val))

    def getFlowIntoNode(self, n):
        if n == self.startNode:
            return -self.Q
        else:
            return self.Q


class PipeNetwork():
    def __init__(self, Pipes=None, Loops=None, Nodes=None, fluid=None):
        self.pipes = Pipes if Pipes else []
        self.loops = Loops if Loops else []
        self.nodes = Nodes if Nodes else []
        self.Fluid = fluid if fluid else Fluid()

    def buildNodes(self):
        for p in self.pipes:
            if not self.nodeBuilt(p.startNode):
                self.nodes.append(Node(p.startNode, self.getNodePipes(p.startNode)))
            if not self.nodeBuilt(p.endNode):
                self.nodes.append(Node(p.endNode, self.getNodePipes(p.endNode)))

    def nodeBuilt(self, node):
        for n in self.nodes:
            if n.name == node:
                return True
        return False

    def getNodePipes(self, node):
        return [p for p in self.pipes if p.oContainsNode(node)]

    def getNode(self, name):
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def getPipe(self, name):
        for p in self.pipes:
            if p.Name() == name:
                return p
        return None

    def getNodeFlowRates(self):
        return [n.getNetFlowRate() for n in self.nodes]

    def getLoopHeadLosses(self):
        return [loop.getLoopHeadLoss() for loop in self.loops]

    def findFlowRates(self):
        N = len(self.nodes) + len(self.loops)
        Q0 = np.full(N, 10.0)

        def fn(q):
            for i, pipe in enumerate(self.pipes):
                pipe.Q = q[i]
            L = self.getNodeFlowRates()
            L += self.getLoopHeadLosses()
            return L

        fsolve(fn, Q0)
        return

    def getNodePressures(self, knownNode, knownNodeP):
        for nd in self.nodes:
            nd.P = 0.0
            nd.oCalculated = False
        for l in self.loops:
            if not l.pipes:
                continue
            startNodeName = l.pipes[0].startNode
            currentP = self.getNode(startNodeName).P
            for p in l.pipes:
                phl = p.getFlowHeadLoss(startNodeName)
                currentP -= phl
                if startNodeName == p.startNode:
                    startNodeName = p.endNode
                else:
                    startNodeName = p.startNode
                nd = self.getNode(startNodeName)
                nd.P = currentP
        nRef = self.getNode(knownNode)
        deltaP = knownNodeP - nRef.P
        for nd in self.nodes:
            nd.P += deltaP

    def printPipeFlowRates(self, SI=True):
        for p in self.pipes:
            p.printPipeFlowRate(SI=SI)

    def printNetNodeFlows(self, SI=True):
        for nd in self.nodes:
            Q = nd.QNet if SI else nd.QNet * UC.L_to_ft3
            units = "L/s" if SI else "cfs"
            print(f"net flow into node {nd.name} is {Q:.2f} ({units})")

    def printLoopHeadLoss(self, SI=True):
        cf = UC.m_to_psi(1, self.Fluid.rho)
        units = "m of water" if SI else "psi"
        for l in self.loops:
            hl = l.getLoopHeadLoss()
            val = hl if SI else hl * cf
            print(f"head loss for loop {l.name} is {val:.2f} ({units})")

    def printPipeHeadLoss(self, SI=True):
        for p in self.pipes:
            p.printPipeHeadLoss(SI=SI)

    def printNodePressures(self, SI=True):
        if SI:
            for nd in self.nodes:
                print(f"Pressure at node {nd.name} = {nd.P:.2f} m of water")
        else:
            cf = UC.m_to_psi(1, self.Fluid.rho)
            for nd in self.nodes:
                print(f"Pressure at node {nd.name} = {nd.P * cf:.2f} psi")


def main():
    # Using English units: SIUnits = False
    SIUnits = False
    # Create water (room temp water in English units)
    water = Fluid(mu=20.50e-6, rho=62.3, SI=False)
    # Roughness values in ft:
    r_CI = 0.00085  # cast iron
    r_CN = 0.003  # concrete
    PN = PipeNetwork(fluid=water)
    # Add pipes: Length (in inches), Diameter (in inches), roughness (in ft)
    PN.pipes.append(Pipe('a', 'b', 1000, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('a', 'h', 1600, 24, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('b', 'c', 500, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('b', 'e', 800, 16, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('c', 'd', 500, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('c', 'f', 800, 16, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('d', 'g', 800, 16, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('e', 'f', 500, 12, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('e', 'i', 800, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('f', 'g', 500, 12, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('g', 'j', 800, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('h', 'i', 1000, 24, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('i', 'j', 1000, 24, r_CN, water, SI=SIUnits))
    PN.buildNodes()
    # Set external flows (in cfs)
    PN.getNode('h').setExtFlow(+10, SI=False)
    PN.getNode('e').setExtFlow(-3, SI=False)
    PN.getNode('f').setExtFlow(-5, SI=False)
    PN.getNode('d').setExtFlow(-2, SI=False)
    PN.loops.append(
        Loop('A', [PN.getPipe('a-b'), PN.getPipe('b-e'),
                   PN.getPipe('e-i'), PN.getPipe('h-i'), PN.getPipe('a-h')])
    )
    PN.loops.append(
        Loop('B', [PN.getPipe('b-c'), PN.getPipe('c-f'),
                   PN.getPipe('e-f'), PN.getPipe('b-e')])
    )
    PN.loops.append(
        Loop('C', [PN.getPipe('c-d'), PN.getPipe('d-g'),
                   PN.getPipe('f-g'), PN.getPipe('c-f')])
    )
    PN.loops.append(
        Loop('D', [PN.getPipe('e-i'), PN.getPipe('i-j'),
                   PN.getPipe('g-j'), PN.getPipe('d-g')])
    )
    PN.findFlowRates()
    knownP_m = UC.psi_to_m(80, water.rho)
    PN.getNodePressures(knownNode='h', knownNodeP=knownP_m)
    PN.printPipeFlowRates(SI=SIUnits)
    print()
    print("Check node flows:")
    PN.printNetNodeFlows(SI=SIUnits)
    print()
    print("Check loop head loss:")
    PN.printLoopHeadLoss(SI=SIUnits)
    print()
    PN.printPipeHeadLoss(SI=SIUnits)
    print()
    PN.printNodePressures(SI=SIUnits)


if __name__ == "__main__":
    main()
