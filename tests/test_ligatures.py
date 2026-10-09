from sophons.parsers.ligatures import repair_ligatures


def test_private_use_glyphs_are_spelled_out_from_the_words_they_make():
    pages = [
        "The Appellant led suit and was awarded ve months’ salary.",
        "The sta eort was sucient; the Ocer saw no conict. It was justied.",
    ]
    repaired, changed = repair_ligatures(pages)
    assert changed
    assert repaired[0] == "The Appellant filed suit and was awarded five months’ salary."
    assert repaired[1] == (
        "The staff effort was sufficient; the Officer saw no conflict. It was justified."
    )


def test_standard_ligature_characters_are_spelled_out():
    repaired, changed = repair_ligatures(["Notiﬁcation of the ﬂoor staﬀ oﬃce"])
    assert changed and repaired == ["Notification of the floor staff office"]


def test_a_glyph_is_left_alone_without_clear_evidence():
    # No spelling makes a known word, so nothing is guessed.
    unknown = ["Symbol  marks the entry", "xy"]
    assert repair_ligatures(unknown) == (unknown, False)
    # Two spellings are equally good ("fine" and "fl..." no; "fit"/"flit"): a tie is not settled.
    plain = ["Nothing to repair here."]
    assert repair_ligatures(plain) == (plain, False)


def test_a_word_with_two_glyphs_is_settled_once_one_is_known():
    pages = ["rst the oce led it; the nal ocer certied the adavit."]
    repaired, _ = repair_ligatures(pages)
    assert repaired == ["first the office filed it; the final officer certified the affidavit."]
