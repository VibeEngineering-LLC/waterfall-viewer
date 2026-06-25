import pytest
from awf.io.nuclide_lib import Nuclide, GammaLine, default_library
from awf.io.nuclide_categories import (category_of, classify_lifetime_seconds,
  classify_lifetime, enrich_nuclide, enrich_all, nuclides_by_category,
  nuclides_by_lifetime, threshold_seconds, threshold_days, CATEGORIES, LIFETIMES)

def _nuc(name, hv=None, hu=None):
    return Nuclide(name=name, half_life_value=hv,
                   half_life_unit=hu, lines=(GammaLine(100.0, 50.0),))

def test_categories_constant():
    assert CATEGORIES == ("natural", "technogenic", "medical", "fission")
    assert LIFETIMES == ("short", "long")

def test_category_of_known():
    assert category_of("Cs-137") == "fission"
    assert category_of("K-40") == "natural"
    assert category_of("Co-60") == "technogenic"
    assert category_of("Tc-99m") == "medical"

def test_category_of_unknown():
    assert category_of("Zz-999") is None

def test_threshold():
    assert threshold_days() == pytest.approx(60.0)
    assert threshold_seconds() == pytest.approx(60 * 86400.0)

def test_classify_lifetime_seconds():
    thr = threshold_seconds()
    assert classify_lifetime_seconds(thr - 1.0) == "short"
    assert classify_lifetime_seconds(thr) == "long"
    assert classify_lifetime_seconds(thr + 1.0) == "long"
    assert classify_lifetime_seconds(None) is None

def test_classify_lifetime_nuclide():
    assert classify_lifetime(_nuc("Cs-137", 30.0, "year")) == "long"
    assert classify_lifetime(_nuc("I-131", 8.0, "day")) == "short"
    assert classify_lifetime(_nuc("X")) is None

def test_enrich_nuclide():
    nuc = _nuc("Cs-137", 30.0, "year")
    e = enrich_nuclide(nuc)
    assert e.category == "fission"
    assert e.lifetime == "long"
    assert e.name == "Cs-137"
    assert e.lines is nuc.lines
    assert e is not nuc

def test_enrich_all_and_filters():
    nucs = enrich_all([_nuc("Cs-137", 30.0, "year"),
                       _nuc("K-40", 1.25e9, "year"),
                       _nuc("I-131", 8.0, "day"),
                       _nuc("Zz-999", 1.0, "day")])
    
    names = lambda lst: {n.name for n in lst}
    
    assert names(nuclides_by_category(nucs, "fission")) >= {"Cs-137", "I-131"}
    assert "K-40" not in names(nuclides_by_category(nucs, "fission"))
    
    assert names(nuclides_by_category(nucs, "natural")) >= {"K-40"}
    
    assert names(nuclides_by_lifetime(nucs, "short")) >= {"I-131", "Zz-999"}
    
    assert names(nuclides_by_lifetime(nucs, "long")) >= {"Cs-137", "K-40"}
    
    zz = next(n for n in nucs if n.name == "Zz-999")
    assert zz.category is None

def test_default_library_enrichable():
    lib = enrich_all(default_library())
    try:
        cs137 = next(n for n in lib if n.name == "Cs-137")
        assert cs137.category == "fission"
    except StopIteration:
        # Если Cs-137 нет в библиотеке, просто проверяем, что enrich_all не падает
        assert len(lib) == len(default_library())
