from access import PAGE_ACCESS, PAGES, can_access, get_user, user_options


def test_user_options_use_display_names_with_stable_ids() -> None:
    assert user_options() == {
        "Emma": "111",
        "Max": "222",
        "Paul": "333",
    }


def test_access_rules_allow_expected_pages() -> None:
    pm = get_user("111")
    rm = get_user("222")
    admin = get_user("333")

    assert can_access(pm, "strategies")
    assert can_access(pm, "analytics")
    assert not can_access(pm, "suitability")
    assert not can_access(pm, "admin")

    assert can_access(rm, "suitability")
    assert can_access(rm, "analytics")
    assert not can_access(rm, "strategies")
    assert not can_access(rm, "admin")

    assert all(can_access(admin, page) for page in PAGES)


def test_every_declared_page_has_access_rules() -> None:
    assert set(PAGES) == set(PAGE_ACCESS)
