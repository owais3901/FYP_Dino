from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")


def detect(img_path):
    try:
        TEXT_PROMPT = "chair . person . dog . cat."
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
        # cv2.imwrite("annotated_image.jpg", annotated_frame)
        return boxes,logits,phrases
    except Exception as e:
        print("Exception in detect",e)
        return False,False,False

print(detect(".asset/cats.png"))