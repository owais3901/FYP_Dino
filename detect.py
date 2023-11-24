from groundingdino.util.inference import load_model, load_image, predict,annotate
import cv2

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")


def detect(img_path):
    try:
        TEXT_PROMPT = "chair . person . dog . cat. shoe"
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

        image_source, image = load_image(img_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device = "cpu"
        )
        bbox_values = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        print(bbox_values)
        # cv2.imwrite("annotated_image.jpg", annotated_frame)
        return bbox_values,logits,phrases
    except Exception as e:
        print("Exception in detect",e)
        return None,None,None