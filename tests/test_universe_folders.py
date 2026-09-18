"""Universe folders: bundled sector/theme defaults plus user-managed nesting."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from alphalineage.api.app import app, get_panel
from alphalineage.data.universe import (
    bundled_classification,
    bundled_snapshot_name,
    bundled_snapshot_specs,
    bundled_universe,
)


@pytest.fixture
def client(synthetic_panel) -> Iterator[TestClient]:
    app.dependency_overrides[get_panel] = lambda: synthetic_panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _tree(client: TestClient) -> dict:
    response = client.get("/universe-folders")
    assert response.status_code == 200, response.text
    return response.json()


def _folder(tree: dict, name: str, parent: str | None = None) -> dict:
    matches = [
        folder
        for folder in tree["folders"]
        if folder["name"] == name and (parent is None or folder["parent"] == parent)
    ]
    assert len(matches) == 1, (name, tree["folders"])
    return matches[0]


def test_sector_and_theme_subsets_are_checksummed_filters_of_the_sp500_snapshot() -> None:
    parent = set(bundled_universe("builtin-sp500-current").all_symbols())
    specs = [item for item in bundled_snapshot_specs() if item.get("parent")]
    sectors = [item for item in specs if item["id"].startswith("builtin-sp500-sector-")]
    assert len(sectors) == 11
    # Sectors partition the parent snapshot exactly: no symbol lost or double counted.
    sector_members = [bundled_universe(item["id"]).all_symbols() for item in sectors]
    assert sum(len(members) for members in sector_members) == len(parent) == 503
    assert set().union(*map(set, sector_members)) == parent

    energy = bundled_universe("sp500-energy")
    assert energy.name == "builtin-sp500-sector-energy"
    assert {"XOM", "CVX", "COP"} <= set(energy.all_symbols())
    assert energy.mode == "static_snapshot"
    assert energy.definition["classification"] == {"sectors": ["Energy"]}
    assert "survivorship" in energy.provenance["note"]

    ag = bundled_universe("builtin-sp500-theme-agriculture-food")
    assert {"ADM", "BG", "CTVA", "CF", "MOS", "DE"} <= set(ag.all_symbols())
    semis = bundled_classification("builtin-sp500-theme-semiconductors")
    assert semis["NVDA"] == {"sector": "Information Technology", "sub_industry": "Semiconductors"}
    assert bundled_snapshot_name("sp500-semiconductors") == "builtin-sp500-theme-semiconductors"
    for item in specs:
        assert set(bundled_universe(item["id"]).all_symbols()) <= parent


def test_bundled_subsets_start_inside_collapsed_default_folders(client) -> None:
    tree = _tree(client)
    sectors = _folder(tree, "S&P 500 sectors (GICS)")
    themes = _folder(tree, "S&P 500 themes")
    technology = _folder(tree, "Technology", themes["id"])
    assert sectors["builtin"] is True and sectors["parent"] is None
    placements = tree["placements"]
    assert placements["builtin-sp500-sector-energy"] == sectors["id"]
    assert placements["builtin-sp500-theme-semiconductors"] == technology["id"]
    # Existing top-level universes stay where users already find them.
    assert placements["sp500-lite"] is None
    assert placements["builtin-sp500-current"] is None

    listed = {item["name"]: item for item in client.get("/universes?summary=true").json()}
    assert listed["builtin-sp500-sector-energy"]["folder_id"] == sectors["id"]
    assert listed["builtin-sp500-theme-semiconductors"]["folder_path"] == [
        "S&P 500 themes",
        "Technology",
    ]
    assert listed["sp500-lite"]["folder_path"] == []


def test_user_can_create_rename_nest_move_and_delete_folders(client) -> None:
    created = client.post("/universe-folders", json={"name": "Research"})
    assert created.status_code == 200, created.text
    research = created.json()
    child = client.post(
        "/universe-folders", json={"name": "Energy ideas", "parent": research["id"]}
    ).json()
    assert child["parent"] == research["id"]

    duplicate = client.post("/universe-folders", json={"name": "research"})
    assert duplicate.status_code == 409

    renamed = client.patch(f"/universe-folders/{child['id']}", json={"name": "Oil & gas"})
    assert renamed.status_code == 200 and renamed.json()["name"] == "Oil & gas"

    # A folder cannot move under its own descendant.
    cycle = client.patch(f"/universe-folders/{research['id']}", json={"parent": child["id"]})
    assert cycle.status_code == 400

    moved = client.put(
        "/universe-folders/placements",
        json={"universes": ["builtin-sp500-sector-energy", "sp500-lite"], "folder": child["id"]},
    )
    assert moved.status_code == 200, moved.text
    tree = _tree(client)
    assert tree["placements"]["builtin-sp500-sector-energy"] == child["id"]
    assert tree["placements"]["sp500-lite"] == child["id"]

    unknown = client.put(
        "/universe-folders/placements", json={"universes": ["nope"], "folder": None}
    )
    assert unknown.status_code == 404

    # Deleting a folder never deletes universes: its contents move up one level.
    assert client.delete(f"/universe-folders/{child['id']}").status_code == 200
    tree = _tree(client)
    assert tree["placements"]["builtin-sp500-sector-energy"] == research["id"]
    assert all(folder["id"] != child["id"] for folder in tree["folders"])

    # Back to the top level.
    client.put(
        "/universe-folders/placements",
        json={"universes": ["builtin-sp500-sector-energy"], "folder": None},
    )
    assert _tree(client)["placements"]["builtin-sp500-sector-energy"] is None


def test_builtin_folders_can_be_renamed_and_deleted_without_losing_members(client) -> None:
    tree = _tree(client)
    themes = _folder(tree, "S&P 500 themes")
    technology = _folder(tree, "Technology", themes["id"])
    renamed = client.patch(f"/universe-folders/{themes['id']}", json={"name": "Industry themes"})
    assert renamed.status_code == 200

    assert client.delete(f"/universe-folders/{technology['id']}").status_code == 200
    tree = _tree(client)
    assert _folder(tree, "Industry themes")["id"] == themes["id"]
    assert tree["placements"]["builtin-sp500-theme-semiconductors"] == themes["id"]
    assert all(folder["id"] != technology["id"] for folder in tree["folders"])


def test_custom_universes_can_be_placed_and_their_placement_is_forgotten_on_delete(client) -> None:
    folder = client.post("/universe-folders", json={"name": "Mine"}).json()
    spec = {
        "name": "folder-test",
        "memberships": [{"symbol": "AAPL", "entry": "2020-01-01", "exit": None}],
    }
    assert client.post("/universes", json=spec).status_code == 200
    assert (
        client.put(
            "/universe-folders/placements",
            json={"universes": ["folder-test"], "folder": folder["id"]},
        ).status_code
        == 200
    )
    assert _tree(client)["placements"]["folder-test"] == folder["id"]
    assert client.delete("/universes/folder-test").status_code == 200
    assert "folder-test" not in _tree(client)["placements"]
    # Re-creating the name starts fresh at the top level instead of inheriting stale state.
    assert client.post("/universes", json=spec).status_code == 200
    assert _tree(client)["placements"]["folder-test"] is None
