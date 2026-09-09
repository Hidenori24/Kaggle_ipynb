import numpy as np

from gh_baggage_core.geometry import item_world_aabb, oriented_dims


def test_oriented_dims_matches_env_convention():
    l, w, h = 0.5, 0.3, 0.2
    assert oriented_dims(l, w, h, 0) == (0.5, 0.3, 0.2)
    assert oriented_dims(l, w, h, 1) == (0.5, 0.2, 0.3)
    assert oriented_dims(l, w, h, 2) == (0.2, 0.3, 0.5)
    assert oriented_dims(l, w, h, 3) == (0.3, 0.5, 0.2)
    assert oriented_dims(l, w, h, 4) == (0.3, 0.2, 0.5)
    assert oriented_dims(l, w, h, 5) == (0.2, 0.5, 0.3)


def test_item_world_aabb_identity_orientation():
    pos = (1.0, 2.0, 3.0)
    orn = (0.0, 0.0, 0.0, 1.0)  # identity quaternion
    lo, hi = item_world_aabb(pos, orn, length=0.4, width=0.6, height=0.2)
    np.testing.assert_allclose(lo, [1.0 - 0.2, 2.0 - 0.3, 3.0 - 0.1])
    np.testing.assert_allclose(hi, [1.0 + 0.2, 2.0 + 0.3, 3.0 + 0.1])


def test_item_world_aabb_z_90_rotation_swaps_xy_extent():
    pos = (0.0, 0.0, 0.0)
    # 90 degree rotation about Z: (x, y, z, w) = (0, 0, sin(45deg), cos(45deg))
    s = 0.7071067811865476
    orn = (0.0, 0.0, s, s)
    lo, hi = item_world_aabb(pos, orn, length=0.4, width=0.6, height=0.2)
    # After a 90deg yaw, the half-extents along x/y should swap (0.4 <-> 0.6)
    np.testing.assert_allclose(hi - lo, [0.6, 0.4, 0.2], atol=1e-9)
