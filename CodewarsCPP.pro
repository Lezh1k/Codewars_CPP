TEMPLATE = app
CONFIG += console c++20
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -Wall -Wextra -pedantic

INCLUDEPATH += inc \

SOURCES += \
        main.cpp \
        src/blocks_number.cpp \
        src/break_piece.cpp \
        src/queens.cpp \
        src/vertex.cpp \
        src/game_model.cpp

HEADERS += \
    inc/blocks_number.h \
    inc/break_piece.h \
    inc/queens.h \
    inc/vetex.h \
    inc/game_model.h \
    inc/game.hpp
