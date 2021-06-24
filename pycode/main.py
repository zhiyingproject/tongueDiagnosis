import cv2
import matplotlib.pyplot as plt
import dlib
import sys
import glob


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    name_label = "N"
    for i_file, file_name in enumerate(glob.glob("../originalData/apparatus/*.jpg")):

        original = cv2.imread(file_name)
        print(original.shape)
        sys.exit(0)
        try:
            resized = cv2.resize(original, (200, 356))
        except:
            print(f"Cannot resize {file_name}")

        gray = cv2.cvtColor(src=resized, code=cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        area = []
        coordinates = []
        for face in faces:
            x1 = face.left() - 3
            y1 = face.top() - 6
            x2 = face.right() + 3
            y2 = face.bottom() + 20
            cv2.rectangle(
                img=resized,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0, 255, 0),
                thickness=4
            )
            area.append((x2 - x1) * (y2 - y1))
            coordinates.append((x1, y1, x2, y2))
        if not area:
            print("Cannot find face!", file_name)
            input("...")
            continue
        idx_max = area.index(max(area))
        x1 = coordinates[idx_max][0]
        y1 = coordinates[idx_max][1]
        x2 = coordinates[idx_max][2]
        y2 = coordinates[idx_max][3]
        print(y1,y2,x1,x2)
        ROI = resized[y1:y2, x1:x2]
        ROI_resize = cv2.resize(ROI, (90, 100))
        print(file_name)
        cv2.imwrite(f"./data/clean/{i_file}.png", ROI_resize)
        '''cv2.imshow(winname="1", mat=ROI)
        cv2.waitKey(
            delay=0
            )
        cv2.destroyAllWindows()
        '''
        '''
        w = int((x2 - x1)/3.0)
        h = int((y2 - y1)/3.0)
        for i in range(0, 3):
            y_sub_1 = y1 + i * h
            y_sub_2 = y1 + (i + 1) * h
            for j in range(0, 3):
                print(i,j,"*******")
                x_sub_1 = x1 + j * w
                x_sub_2 = x1 + (j + 1) * w

                ROI = resized[y_sub_1:y_sub_2, x_sub_1:x_sub_2]
                cv2.imwrite(f"./data{file_name}{i}{j}.png", ROI)

        '''

        # cv2.imshow(winname="1", mat=resized)
        # cv2.waitKey(
        #    delay=0
        #    )
    # cv2.destroyAllWindows()

'''
    plt.imshow(original)
    plt.show()
    plt.imshow(resized)

    plt.show()

    print(original.shape)
    print(resized.shape)
'''
