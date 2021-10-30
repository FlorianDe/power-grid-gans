import pandapower as pp
from pandapower import pandapowerNet
import pandapower.networks as pn
# import pandapower.test

def create_cigre_net() -> pandapowerNet:
    cigre_net = pn.create_cigre_network_mv(with_der=False)
    # cigre_net_with_wind = pn.create_cigre_network_mv(with_der='pv_wind')
    # cigre_net_with_all = pn.create_cigre_network_mv(with_der='all')
    return cigre_net


def create_custom_net() -> pandapowerNet:
    #create empty net
    net = pp.create_empty_network()

    #create buses
    b1 = pp.create_bus(net, vn_kv=20., name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")

    #create bus elements
    pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b3, p_mw=0.1, q_mvar=0.05, name="Load")

    #create branch elements
    tid = pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=0.1, name="Line",std_type="NAYY 4x50 SE")

    return net


if __name__ == "__main__":
    pass
    cigre_net = create_cigre_net()
    custom_net = create_custom_net()
    pp.runpp(cigre_net)

    # pandapower.test.run_all_tests()
