import pandas as pd
import pytest

from clwe_chains.config import load_config
from clwe_chains.pairs import eligible, low_resource, scarce_pairs, select_pairs, widest_hub
from clwe_chains.shared_list import list_languages

LOW = {"mt", "ga", "sw", "eu", "tl", "ur", "et", "kk", "ka", "ta"}


def _inputs():
    cfg = load_config()
    pc = cfg["pairs"]
    langs = pd.read_csv(cfg.path("v0_languages")).set_index("ft_code")
    nodes = list_languages(pd.read_parquet(cfg.path("list") / "shared_list.parquet"))
    family = langs.loc[nodes, "family"].to_dict()
    contact = {frozenset(p) for p in pc["contact"]}
    return nodes, family, pc["sizes_gib"], pc, contact


def test_low_resource_bottom_third():
    pc = load_config()["pairs"]
    nodes = sorted(pc["sizes_gib"])
    assert set(low_resource(pc["sizes_gib"], nodes)) == LOW


def test_eligible():
    family = {"en": "Indo-European", "de": "Indo-European", "ar": "Afro-Asiatic",
              "fa": "Indo-European", "ja": "Japonic"}
    contact = {frozenset(("ar", "fa"))}
    assert not eligible("en", "de", family, contact)
    assert not eligible("ar", "fa", family, contact)
    assert eligible("ar", "ja", family, contact)


@pytest.mark.data
def test_select_pairs_counts_and_rules():
    nodes, family, sizes, pc, contact = _inputs()
    df = select_pairs(nodes, family, sizes, pc, contact, cap=pc["cap_b"], seed=pc["seed"])
    assert len(df) == 40 and (df["stratum"].value_counts() == 20).all()
    directed = set(zip(df["s"], df["t"], strict=True))
    assert {("en", "ar"), ("en", "zh"), ("en", "ja")} <= directed
    assert all(eligible(s, t, family, contact) for s, t in directed)
    assert len({frozenset(p) for p in directed}) == 40
    b = df[df["stratum"] == "b"]
    assert all(s in LOW or t in LOW for s, t in zip(b["s"], b["t"], strict=True))
    assert pd.concat([b["s"], b["t"]]).value_counts().max() <= 3
    a = df[df["stratum"] == "a"]
    assert not any(s in LOW or t in LOW for s, t in zip(a["s"], a["t"], strict=True))
    x = a[a["pool"] == "xling"]
    assert pd.concat([x["s"], x["t"]]).value_counts()[["fi", "tr"]].tolist() == [3, 3]
    assert {"ja", "zh", "fa", "el", "hi"} <= set(a["s"]) | set(a["t"])


@pytest.mark.data
def test_select_pairs_deterministic():
    nodes, family, sizes, pc, contact = _inputs()
    one = select_pairs(nodes, family, sizes, pc, contact, cap=3, seed=0)
    two = select_pairs(nodes, family, sizes, pc, contact, cap=3, seed=0)
    other = select_pairs(nodes, family, sizes, pc, contact, cap=3, seed=1)
    pd.testing.assert_frame_equal(one, two)
    assert not one.equals(other)


def test_scarce_pairs_rule():
    langs = ["lo", "en", "fr", "zh"]
    size = pd.DataFrame(0, index=langs, columns=langs)

    def setsz(a, b, v):
        size.at[a, b] = size.at[b, a] = v
    setsz("lo", "en", 3000)
    setsz("lo", "fr", 900)
    setsz("lo", "zh", 100)
    setsz("en", "zh", 9000)
    setsz("fr", "zh", 9000)
    fam = {"lo": "Uralic", "en": "IE", "fr": "IE", "zh": "ST"}
    out = scarce_pairs(size, fam, ["lo"], ["en", "fr", "zh"], max_direct=500)
    assert out[["s", "t", "direct", "hub", "route"]].values.tolist() == [
        ["lo", "zh", 100, "en", 3000]]


def test_scarce_pairs_drops_same_family_and_own_hub():
    langs = ["lo", "en", "fi"]
    size = pd.DataFrame(0, index=langs, columns=langs)
    size.loc["lo", "en"] = size.loc["en", "lo"] = 100
    size.loc["lo", "fi"] = size.loc["fi", "lo"] = 50
    size.loc["en", "fi"] = size.loc["fi", "en"] = 9000
    fam = {"lo": "Uralic", "en": "IE", "fi": "Uralic"}
    # lo-fi: same family; lo-en: its only possible hub is fi, which gives route 50 (kept);
    # with fi as the only other node the hub isn't the partner, so lo-en stays
    out = scarce_pairs(size, fam, ["lo"], ["en", "fi"], max_direct=500)
    assert out[["s", "t", "hub"]].values.tolist() == [["lo", "en", "fi"]]


def test_widest_hub_excludes_the_endpoints():
    langs = ["en", "fr", "zh", "ja"]
    size = pd.DataFrame(0, index=langs, columns=langs)
    for a, b, v in (("en", "zh", 9000), ("en", "ja", 8000), ("fr", "zh", 3000), ("fr", "ja", 7000),
                    ("zh", "ja", 500), ("en", "fr", 4000)):
        size.at[a, b] = size.at[b, a] = v
    assert widest_hub(size, "zh", "ja", langs) == ("en", 8000)
    assert widest_hub(size, "en", "ja", langs) == ("fr", 4000)   # en is an endpoint: never the hub
