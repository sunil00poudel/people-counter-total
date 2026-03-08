import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# model
model = YOLO("yolo11n.pt")

# gui
root = tk.Tk()
root.title('people counter By nowa')
root.minsize(600, 800)
root.resizable(False, False)

video_label = tk.Label(root)
video_label.pack()

cap = cv2.VideoCapture('datasets/people.mp4')

# Variables for counting
total_count = 0  # Changed 'count' to 'total_count' to match your logic below
counted_id = set()

def update_frame():
    global total_count, counted_id  # Added global to allow updating these variables
    ret, frame = cap.read()
    if ret:
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (600, 800))
        result = model.track(frame, persist=True)

        res = result[0]

        if res.boxes is not None:
            for box in res.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    if box.id is not None:
                        obj_id = int(box.id[0])
                        # Counting Logic
                        if obj_id not in counted_id:
                            total_count += 1
                            counted_id.add(obj_id)

                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = coords

                        # Drawing Logic
                        if x1 >= 0 and y1 >= 0 and x2 <= 600 and y2 <= 800:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y1) - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display Total Count on Frame (Top Right)
        cv2.putText(frame, f"Total People: {total_count}", (350, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.config(image=img_tk)
        video_label.img_tk = img_tk  # type:ignore

    video_label.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()