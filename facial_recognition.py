#!/usr/bin/env python
import os
import numpy as np
import fnmatch
import cv2 as cv
import sys
import datetime

def process_image(img):
    faces = detect_faces(img)
    if faces == ():
        cv.destroyWindow('faces')
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.resizeWindow('img', 600, 600)
        cv.imshow('img', img)
    else:
        # show_box_around_faces(img, faces)
        show_detected_faces(img, faces)
        name_labels = face_recognition_scores(img, faces, data)
        show_names(img, faces, name_labels)
    return faces

def detect_faces(img):
    face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                          flags=cv.CASCADE_SCALE_IMAGE)
    return faces

def show_box_around_faces(img, faces):
    img_copy = img.copy()
    for (x, y, w, h) in faces:
        # img.copyTo(img_copy)
        cv.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.resizeWindow('img', 600, 600)
        cv.imshow('img', img_copy)

def show_detected_faces(img, faces):
    size = 100
    faces_stacked = np.zeros([size,1,3], dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv.namedWindow('faces', cv.WINDOW_NORMAL)
        new_face = img[y:y + h, x:x + w]
        new_face = cv.resize(new_face, dsize=(size, size))
        faces_stacked = np.hstack((faces_stacked, new_face))
    cv.imshow('faces', faces_stacked)

def show_names(img, faces, names):
    img_copy = img.copy()
    for (x, y, w, h), name_label in zip(faces, names):
        cv.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(img_copy, name_label, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 3,
                   (255, 0, 0), 6, cv.LINE_AA)
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.resizeWindow('img', 600, 600)
        cv.imshow('img', img_copy)


def create_database(directory):
    '''
    Process all images in the given directory.
    Every image is cropped to the detected face, resized to 100x100 and save in another directory (orignal directory name + "2").

    @param directory:    directory to process
    @param show:         bool, show all intermediate results

    '''
    # load a pre-trained classifier
    cascade = cv.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    # loop through all files
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_in = directory + "/" + filename
            file_out = directory + "_normalized/" + filename
            img = cv.imread(file_in)
            print file_in
            # do face detection
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # gray = cv.equalizeHist(gray)
            # rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            #                                  flags=cv.CASCADE_SCALE_IMAGE)  # cv.CV_HAAR_SCALE_IMAGE)
            # print rects
            rects = detect_faces(img)
            if rects == ():
                print "ERROR: " + file_in
                continue
            rects[:, 2:] += rects[:, :2]
            rects = rects[0, :]

            # crop image to the rectangle and resample it to 100x100 pixels
            resultTemp = gray[rects[1]:rects[3], rects[0]:rects[2]]
            result = cv.resize(resultTemp, (100, 100))  # TODO

            cv.imwrite(file_out, result)
    cv.destroyAllWindows()

def createX(directory,nbDim=10000):
    '''
    Create an array that contains all the images in directory.
    @return np.array, shape=(nb images in directory, nb pixels in image)
    '''
    # filenames = fnmatch.filter(os.listdir(directory),'*.jpg')
    filenames = os.listdir(directory)
    nbImages = len(filenames)
    X = np.zeros( (nbImages,nbDim) )#, dtype=np.uint8 )
    for i,filename in enumerate( filenames ):
        file_in = directory+"/"+filename
        img = cv.imread(file_in)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # gray = cv.equalizeHist(gray)
        X[i,:] = gray.flatten()
    print X.dtype
    return X


def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    return W.transpose().dot(X - mu)  # TODO


def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    return W.dot(Y) + mu  # TODO


def pca(X, nb_components=0):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n, d] = X.shape
    if (nb_components <= 0) or (nb_components > n):
        nb_components = n
    print n
    print nb_components

    # TODO
    mu = X.mean(0)
    X -= mu
    cov_mat = X.dot(X.transpose()).dot(1. / (n - 1))
    print cov_mat.shape
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    print eig_val.shape
    print eig_vec.shape

    eig = [list(x) for x in zip(eig_val, eig_vec.transpose())]
    eig.sort(key=lambda tup: tup[0], reverse=True)
    eig = eig[:nb_components]
    eig_val, eig_vec = zip(*eig)
    print np.array(eig_vec)[:, 0:nb_components]
    eig_val = np.array(eig_val).reshape(nb_components)
    eig_vec = np.array(eig_vec).reshape(nb_components, nb_components)
    eig_vec = eig_vec.transpose()
    print eig_vec.shape

    eig_vec = X.transpose().dot(eig_vec)
    print eig_vec.shape

    for i in eig_vec.transpose():
        i /= np.linalg.norm(i)

    print eig_vec.shape

    return eig_val, eig_vec, mu

def normalize(img):
    '''
    Normalize an image such that it min=0 , max=255 and type is np.uint8
    '''
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)

def face_recognition_scores(img_test, faces, data):
    gray = cv.cvtColor(img_test, cv.COLOR_BGR2GRAY)
    # gray = cv.equalizeHist(gray)
    name_list = []
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        # crop image to the rectangle and resample it to 100x100 pixels
        result = cv.resize(face, (100, 100))  # TODO
        # cv.imshow('img', result)
        # cv.waitKey(0)
        X = result.flatten()

        [eigenvalues_arnold, eigenvectors_arnold, mu_arnold] = data[0,:]
        [eigenvalues_barack, eigenvectors_barack, mu_barack] = data[1,:]
        [eigenvalues_carlos, eigenvectors_carlos, mu_carlos] = data[2,:]
        [eigenvalues_hung, eigenvectors_hung, mu_hung] = data[3, :]
        [eigenvalues_khanh_duy_nguyen, eigenvectors_khanh_duy_nguyen, mu_khanh_duy_nguyen] = data[4,:]
        [eigenvalues_le_thanh_tri, eigenvectors_le_thanh_tri, mu_le_thanh_tri] = data[5, :]
        [eigenvalues_miguel, eigenvectors_miguel, mu_miguel] = data[6, :]
        [eigenvalues_nguyen_van_truong, eigenvectors_nguyen_van_truong, mu_nguyen_van_truong] = data[7, :]
        [eigenvalues_simon, eigenvectors_simon, mu_simon] = data[8, :]
        [eigenvalues_tran_tien_hung, eigenvectors_tran_tien_hung, mu_tran_tien_hung] = data[9, :]
        [eigenvalues_trinh_chung, eigenvectors_trinh_chung, mu_trinh_chung] = data[10, :]
        [eigenvalues_trong_nguyen, eigenvectors_trong_nguyen, mu_trong_nguyen] = data[11, :]

        Y_arnold = project(eigenvectors_arnold, X, mu_arnold)
        X_arnold = reconstruct(eigenvectors_arnold, Y_arnold, mu_arnold)

        Y_barack = project(eigenvectors_barack, X, mu_barack)
        X_barack = reconstruct(eigenvectors_barack, Y_barack, mu_barack)

        Y_carlos = project(eigenvectors_carlos, X, mu_carlos)
        X_carlos = reconstruct(eigenvectors_carlos, Y_carlos, mu_carlos)

        Y_hung = project(eigenvectors_hung, X, mu_hung)
        X_hung = reconstruct(eigenvectors_hung, Y_hung, mu_hung)

        Y_khanh_duy_nguyen = project(eigenvectors_khanh_duy_nguyen, X, mu_khanh_duy_nguyen)
        X_khanh_duy_nguyen = reconstruct(eigenvectors_khanh_duy_nguyen, Y_khanh_duy_nguyen, mu_khanh_duy_nguyen)

        Y_le_thanh_tri = project(eigenvectors_le_thanh_tri, X, mu_le_thanh_tri)
        X_le_thanh_tri = reconstruct(eigenvectors_le_thanh_tri, Y_le_thanh_tri, mu_le_thanh_tri)

        Y_miguel = project(eigenvectors_miguel, X, mu_miguel)
        X_miguel = reconstruct(eigenvectors_miguel, Y_miguel, mu_miguel)

        Y_nguyen_van_truong = project(eigenvectors_nguyen_van_truong, X, mu_nguyen_van_truong)
        X_nguyen_van_truong = reconstruct(eigenvectors_nguyen_van_truong, Y_nguyen_van_truong, mu_nguyen_van_truong)

        Y_simon = project(eigenvectors_simon, X, mu_simon)
        X_simon = reconstruct(eigenvectors_simon, Y_simon, mu_simon)

        Y_tran_tien_hung = project(eigenvectors_tran_tien_hung, X, mu_tran_tien_hung)
        X_tran_tien_hung = reconstruct(eigenvectors_tran_tien_hung, Y_tran_tien_hung, mu_tran_tien_hung)

        Y_trinh_chung = project(eigenvectors_trinh_chung, X, mu_trinh_chung)
        X_trinh_chung = reconstruct(eigenvectors_trinh_chung, Y_trinh_chung, mu_trinh_chung)

        Y_trong_nguyen = project(eigenvectors_trong_nguyen, X, mu_trong_nguyen)
        X_trong_nguyen = reconstruct(eigenvectors_trong_nguyen, Y_trong_nguyen, mu_trong_nguyen)

        score_arnold = np.linalg.norm(X_arnold - X)
        score_barack = np.linalg.norm(X_barack - X)
        score_carlos = np.linalg.norm(X_carlos - X)
        score_hung = np.linalg.norm(X_hung - X)
        score_khanh_duy_nguyen = np.linalg.norm(X_khanh_duy_nguyen - X)
        score_le_thanh_tri = np.linalg.norm(X_le_thanh_tri - X)
        score_miguel = np.linalg.norm(X_miguel - X)
        score_nguyen_van_truong = np.linalg.norm(X_nguyen_van_truong - X)
        score_simon = np.linalg.norm(X_simon - X)
        score_tran_tien_hung = np.linalg.norm(X_tran_tien_hung - X)
        score_trinh_chung = np.linalg.norm(X_trinh_chung - X)
        score_trong_nguyen = np.linalg.norm(X_trong_nguyen - X)

        scores = np.array([score_arnold, score_barack, score_carlos, score_hung,
                           score_khanh_duy_nguyen, score_le_thanh_tri, score_miguel, score_nguyen_van_truong,
                           score_simon, score_tran_tien_hung, score_trinh_chung, score_trong_nguyen])
        best_index = np.argmin(scores)

        print "SCORES:"
        if best_index == 0:
            best = "ARNOLD"
        elif best_index == 1:
            best = "BARACK"
        elif best_index == 2:
            best = "CARLOS"
        elif best_index == 3:
            best = "HUNG"
        elif best_index == 4:
            best = "KHANH DUY NGUYEN"
        elif best_index == 5:
            best = "LE THANH TRI"
        elif best_index == 6:
            best = "MIGUEL"
        elif best_index == 7:
            best = "NGUYEN VAN TRONG"
        elif best_index == 8:
            best = "SIMON"
        elif best_index == 9:
            best = "TRAN TIEN HUNG"
        elif best_index == 10:
            best = "TRINH CHUNG"
        elif best_index == 11:
            best = "TRONG NGUYEN"
        print scores, best
        name_list.append(best)
    return name_list

def record_faces():
    cap = cv.VideoCapture(0)
    while (True):
        ret, img = cap.read()
        face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                              flags=cv.CASCADE_SCALE_IMAGE)
        size = 100
        for (x, y, w, h) in faces:
            new_face = img[y:y + h, x:x + w]
            new_face = cv.resize(new_face, dsize=(size, size))
            filename_new_face = '{date:%Y-%m-%d %H:%M:%S.%f}.jpg'.format(date=datetime.datetime.now())
            cv.imwrite(filename_new_face, img)
            print "New image written: ", filename_new_face

if __name__ == '__main__':
    # for directory in ["data/arnold", "data/barack", "data/carlos",
    #                   "data/khanh_duy_nguyen", "data/le_thanh_tri", "data/miguel",
    #                   "data/tran_tien_hung", "data/trong_nguyen"]:
    #     create_database(directory)
    # create_database("data/hung")
    # sys.exit(0)

    # record_faces()
    # sys.exit(0)

    # create big X arrays for arnold and barack
    X_arnold = createX("data/arnold_normalized")
    X_barack = createX("data/barack_normalized")
    X_carlos = createX("data/carlos_normalized")
    X_hung = createX("data/hung_normalized")
    X_khanh_duy_nguyen = createX("data/khanh_duy_nguyen_normalized")
    X_le_thanh_tri = createX("data/le_thanh_tri_normalized")
    X_miguel = createX("data/miguel_normalized")
    X_nguyen_van_truong = createX("data/nguyen_van_truong_normalized")
    X_simon = createX("data/simon_normalized")
    X_tran_tien_hung = createX("data/tran_tien_hung_normalized")
    X_trinh_chung = createX("data/trinh_chung_normalized")
    X_trong_nguyen = createX("data/trong_nguyen_normalized")

    # do pca
    nb_components = 0
    # [eigenvalues_arnold, eigenvectors_arnold, mu_arnold] = pca(X_arnold, nb_components)
    # [eigenvalues_barack, eigenvectors_barack, mu_barack] = pca(X_barack, nb_components)
    # [eigenvalues_carlos, eigenvectors_carlos, mu_carlos] = pca(X_carlos, nb_components)
    # [eigenvalues_miguel, eigenvectors_miguel, mu_miguel] = pca(X_miguel, nb_components)

    data_arnold = pca(X_arnold, nb_components)
    data_barack = pca(X_barack, nb_components)
    data_carlos = pca(X_carlos, nb_components)
    data_hung = pca(X_hung, nb_components)
    data_khanh_duy_nguyen = pca(X_khanh_duy_nguyen, nb_components)
    data_le_thanh_tri = pca(X_le_thanh_tri, nb_components)
    data_miguel = pca(X_miguel, nb_components)
    data_nguyen_van_truong = pca(X_nguyen_van_truong, nb_components)
    data_simon = pca(X_simon, nb_components)
    data_tran_tien_hung = pca(X_tran_tien_hung, nb_components)
    data_trinh_chung = pca(X_trinh_chung, nb_components)
    data_trong_nguyen = pca(X_trong_nguyen, nb_components)

    data = np.array([data_arnold, data_barack, data_carlos, data_hung, data_khanh_duy_nguyen,
                     data_le_thanh_tri, data_miguel, data_nguyen_van_truong, data_simon,
                     data_tran_tien_hung, data_trinh_chung, data_trong_nguyen])
    # visualize
    if False:
        cv.imshow('img', np.hstack((mu_arnold.reshape(100, 100),
                                     normalize(eigenvectors_arnold[:, 0].reshape(100, 100)),
                                     normalize(eigenvectors_arnold[:, 1].reshape(100, 100)),
                                     normalize(eigenvectors_arnold[:, 2].reshape(100, 100)))
                                    ).astype(np.uint8))
        cv.waitKey(0)
        cv.imshow('img', np.hstack((mu_barack.reshape(100, 100),
                                     normalize(eigenvectors_barack[:, 0].reshape(100, 100)),
                                     normalize(eigenvectors_barack[:, 1].reshape(100, 100)),
                                     normalize(eigenvectors_barack[:, 2].reshape(100, 100)))
                                    ).astype(np.uint8))
        cv.waitKey(0)
        cv.imshow('img', np.hstack((mu_miguel.reshape(100, 100),
                                    normalize(eigenvectors_miguel[:, 0].reshape(100, 100)),
                                    normalize(eigenvectors_miguel[:, 1].reshape(100, 100)),
                                    normalize(eigenvectors_miguel[:, 2].reshape(100, 100)))
                                   ).astype(np.uint8))
        cv.waitKey(0)

    webcam = False
    if webcam:
        cap = cv.VideoCapture(0)
        while (True):
            ret, img = cap.read()
            faces = process_image(img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        for i in range(15,18):
            # img = cv.imread('data/' + str(i) + '.jpg')
            img = cv.imread('data/5.jpg')
            # img = cv.resize(img, dsize=(0,0), fx=0.6, fy=0.6)
            faces = process_image(img)
            cv.waitKey(0)

    # When everything done, release the capture
    if webcam:
        cap.release()
    cv.destroyAllWindows()