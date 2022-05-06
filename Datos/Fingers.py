import cv2
import mediapipe as mp
import os

# carpeta donde guardo entrenamiento
name = "letra_i"
placement = r"C:\Users\Denisse\Hand-Signs-reader\Datos\Imagenes\Training"
folder = placement + "/" + name

#contador para el nombre de las imagenes de mi camara
if not os.path.exists(folder):
    print("Ypur folder has been created:", folder )
    os.makedirs(folder)

#contador para las fotos
cont=0

#inicializo la camara

caption = cv2.VideoCapture(0)


 #almaceno las detecciones que realice mi camara detecta y sigue

clase_manos = mp.solutions.hands
manos = clase_manos.Hands() # Detector de manos

mp_drawing = mp.solutions.drawing_utils#puntos de las manos

while(1):
    ret, frame = caption.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    result = manos.process(color)
    positions = [] #coordenadas

    if result.multi_hand_landmarks:
        for mano in result.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark): 
                height, width, c = frame.shape  
                corx, cory = int(lm.x * width), int(lm.y * height)  
                positions.append([id, corx, cory])
                mp_drawing.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(positions) != 0:
                punto_i1 = positions[4]
                punto_i2 = positions[20]
                punto_i3 = positions[12]
                punto_i4 = positions[0]
                punto_i5 = positions[9]
                x1, y1 = (punto_i5[1] - 100), (punto_i5[2] - 100) #Se obtiene el punto inicial y las longitudes
                width, height = (x1 + 220), (y1 + 220)
                x2, y2 = x1 + width, y1 + height
                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            dedos_reg = cv2.resize(dedos_reg, (220, 220), interpolation=cv2.INTER_CUBIC) #Redimensionamos las fotos
            cv2.imwrite(folder + "/Dedos_{}.jpg".format(cont), dedos_reg)
            cont = cont + 1

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:
        break
caption.release()
cv2.destroyAllWindows()