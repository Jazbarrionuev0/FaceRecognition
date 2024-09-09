import threading
import cv2
from deepface import DeepFace
DeepFace.build_model("VGG-Face")



cap = cv2.VideoCapture(0)


cap.set(3, 1280)
cap.set(4, 720)


counter = 0

face_match = False

reference_img = cv2.imread("reference.jpg")

def check_face(frame):
    global face_match
    try:
        print("Checking face...")
        result = DeepFace.verify(frame, reference_img.copy())
        if result['verified']:
            print("Face matched!")
            face_match = True
        else:
            print("Face not matched.")
            face_match = False
    except ValueError as e:
        print(f"Error during verification: {e}")
        face_match = False


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
