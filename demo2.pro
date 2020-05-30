#-------------------------------------------------
#
# Project created by QtCreator 2020-05-22T01:56:44
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = sift
TEMPLATE = app

INCLUDEPATH += /usr/local/include/opencv4
LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui
LIBS += -lopencv_calib3d
LIBS += -lopencv_core
LIBS += -lopencv_features2d
LIBS += -lopencv_flann
LIBS += -lopencv_highgui
LIBS += -lopencv_imgcodecs
LIBS += -lopencv_imgproc
LIBS += -lopencv_ml
LIBS += -lopencv_objdetect
LIBS += -lopencv_photo
LIBS += -lopencv_shape
LIBS += -lopencv_stitching
LIBS += -lopencv_superres
LIBS += -lopencv_video
LIBS += -lopencv_videoio
LIBS += -lopencv_videostab

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
    SIFT.cpp

HEADERS += \
        mainwindow.h \
    stdafx.h \
    targetver.h \
    descriptor.h \
    keypoint.h \
    keypoint.h \
    SIFT.h


FORMS += \
        mainwindow.ui

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../openCV/opencv-4.3.0/build/lib/release/ -lopencv_imgcodecs
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../openCV/opencv-4.3.0/build/lib/debug/ -lopencv_imgcodecs
else:unix: LIBS += -L$$PWD/../../openCV/opencv-4.3.0/build/lib/ -lopencv_imgcodecs

INCLUDEPATH += $$PWD/../../openCV/opencv-4.3.0/build/include
DEPENDPATH += $$PWD/../../openCV/opencv-4.3.0/build/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../openCV/opencv-4.3.0/build/lib/release/ -lopencv_highgui
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../openCV/opencv-4.3.0/build/lib/debug/ -lopencv_highgui
else:unix: LIBS += -L$$PWD/../../openCV/opencv-4.3.0/build/lib/ -lopencv_highgui

INCLUDEPATH += $$PWD/../../openCV/opencv-4.3.0/build/include
DEPENDPATH += $$PWD/../../openCV/opencv-4.3.0/build/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../openCV/opencv-4.3.0/build/lib/release/ -lopencv_core
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../openCV/opencv-4.3.0/build/lib/debug/ -lopencv_core
else:unix: LIBS += -L$$PWD/../../openCV/opencv-4.3.0/build/lib/ -lopencv_core

INCLUDEPATH += $$PWD/../../openCV/opencv-4.3.0/build/include
DEPENDPATH += $$PWD/../../openCV/opencv-4.3.0/build/include
