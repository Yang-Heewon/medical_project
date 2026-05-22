from vision_rag_cxr.data.labeler_chexbert import CheXbertLikeLabeler


def test_keyword_labeler_detects_pneumothorax():
    labeler = CheXbertLikeLabeler()
    labels = labeler.label_texts(["There is a small right pneumothorax."])[0]
    assert labels["Pneumothorax"] == 1
