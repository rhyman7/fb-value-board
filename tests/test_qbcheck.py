import pandas as pd

from nflmodel import qbcheck


def test_compare_names_loosely_and_flags_differences():
    board = pd.DataFrame({"game_id": ["g1", "g1", "g2", "g2"], "team": ["KC", "DEN", "LV", "NYJ"],
                          "board_qb": ["Patrick Mahomes", "Bo Nix", "Geno Smith", "Justin Fields"],
                          "kickoff": ["2026-11-08T18:00Z"] * 4})
    nfelo = pd.DataFrame({"game_id": ["x", "x", "y"], "date": ["2026-11-08"] * 3, "team": ["KC", "DEN", "LV"],
                          "nfelo_qb": ["Patrick Mahomes II", "Jarrett Stidham", "Geno Smith"]})
    m = qbcheck.compare(board, nfelo).set_index("team")
    assert m.loc["KC", "agree"] and m.loc["LV", "agree"]
    assert m.loc["DEN", "agree"] is False or m.loc["DEN", "agree"] == False  # noqa: E712
    assert m.loc["NYJ", "agree"] is None


def test_apply_writes_override_but_keeps_hand_rows(tmp_path, monkeypatch):
    path = tmp_path / "overrides.csv"
    pd.DataFrame([{"game_id": "g2", "team": "LV", "kind": "starter", "player_id": "00-1", "name": "Aidan O'Connell",
                   "status": "", "source": "team site Nov 1"}]).to_csv(path, index=False)
    qbs = pd.DataFrame({"player_id": ["00-2", "00-3", "00-4"], "name": ["Jarrett Stidham", "Geno Smith", "Jarrett Stidham"],
                        "teams": [{"DEN"}, {"LV"}, {"CHI"}]})
    qbs["key"] = qbs["name"].map(qbcheck.norm)
    monkeypatch.setattr(qbcheck, "qb_ids", lambda season: qbs)
    monkeypatch.setattr(qbcheck, "OUT", tmp_path)
    res = {"checked_utc": "2026-11-05T17:00:00Z", "disagree": [
        {"game_id": "g1", "team": "DEN", "board": "Bo Nix", "nfelo": "Jarrett Stidham", "kickoff": "x"},
        {"game_id": "g2", "team": "LV", "board": "Aidan O'Connell", "nfelo": "Geno Smith", "kickoff": "x"},
        {"game_id": "g3", "team": "NYJ", "board": "Justin Fields", "nfelo": "Nobody Known", "kickoff": "x"}]}
    out = qbcheck.apply(res, 2026, path)
    o = pd.read_csv(path, dtype=str)
    assert [a["team"] for a in out["applied"]] == ["DEN"]
    assert o[o.team == "DEN"].player_id.item() == "00-2"          # the Stidham who has thrown for DEN
    assert o[o.team == "LV"].player_id.item() == "00-1"           # hand row kept
    assert {k["team"] for k in out["skipped"]} == {"LV", "NYJ"}
    # nfelo changes its mind: the auto row is replaced, not duplicated
    res2 = {"checked_utc": "2026-11-06T17:00:00Z", "disagree": [
        {"game_id": "g1", "team": "DEN", "board": "Jarrett Stidham", "nfelo": "Geno Smith", "kickoff": "x"}]}
    qbs.loc[qbs.name == "Geno Smith", "teams"] = [{"DEN", "LV"}]
    qbcheck.apply(res2, 2026, path)
    o = pd.read_csv(path, dtype=str)
    assert (o.team == "DEN").sum() == 1 and o[o.team == "DEN"].player_id.item() == "00-3"


def test_apply_never_starts_a_quarterback_with_an_out_row(tmp_path, monkeypatch):
    path = tmp_path / "overrides.csv"
    pd.DataFrame([{"game_id": "g1", "team": "ARI", "kind": "out", "player_id": "00-9", "name": "Jacoby Brissett",
                   "status": "Out", "source": "traded Nov 4"}]).to_csv(path, index=False)
    qbs = pd.DataFrame({"player_id": ["00-9"], "name": ["Jacoby Brissett"], "teams": [{"ARI"}]})
    qbs["key"] = qbs["name"].map(qbcheck.norm)
    monkeypatch.setattr(qbcheck, "qb_ids", lambda season: qbs)
    monkeypatch.setattr(qbcheck, "OUT", tmp_path)
    res = {"checked_utc": "2026-11-05T17:00:00Z", "disagree": [
        {"game_id": "g1", "team": "ARI", "board": "Gardner Minshew", "nfelo": "Jacoby Brissett", "kickoff": "x"}]}
    out = qbcheck.apply(res, 2026, path)
    assert out["applied"] == [] and out["skipped"][0]["why"].startswith("nfelo's quarterback has an out row")
    assert (pd.read_csv(path, dtype=str).kind == "starter").sum() == 0
