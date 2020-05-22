BIN=Main
OBJ_DIR = objs
MKDIR = mkdir -p

#EIGEN_DIR = /home/badzindagi/dev/eigen

EIGEN_DIR = /usr/include/eigen3/
BULLET_INC_DIR = /usr/local/include/bullet

#BULLET_CMAKE_INC_DIR = /home/badzindagi/dev/bullet3/build_cmake/src
BULLET_CMAKE_INC_DIR = /home/nekokitty/dev/bullet3/build_cmake/src

BULLET_LIB = -L/usr/local/lib/ -lBulletDynamics -lBulletCollision -lLinearMath

GLUT_LIB = -L/home/nekokitty/dev/freeglut-3.2.1/build/lib/ -lglut

PYTHON_INC = /usr/include/python3.8
PYTHON_LIB = /usr/lib/ -lpython3.8

INC = -I./ \
	-I$(EIGEN_DIR) \
	-I$(BULLET_INC_DIR) \
	-I$(BULLET_CMAKE_INC_DIR)

LIBS = -lGLEW -lGL -lGLU ${GLUT_LIB} ${BULLET_LIB} -lm -lstdc++
LIBS_LINK = -lGLEW -lGL -lGLU -lglut -lm -lstdc++


CC = clang++
CFLAGS = -std=c++11 -O3 -Wall -fPIC 

SRC_DIRS = util/ \
	util/json/ \
	anim/ \
	sim/ \
	render/ \
	render/lodepng/ \
	scenes/ \
	scenes/arm/ \
	scenes/particle/ \
	scenes/pendulum/

SRC_FILES = $(BIN).cpp MGSimCore.cpp $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)*.cpp))
OUT_DIRS = $(addprefix $(OBJ_DIR)/, $(SRC_DIRS))
OBJ = $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRC_FILES)))


default: build

build: $(BIN)

print:
	$(OUT_DIRS)

python: $(OUT_DIRS) $(OBJ)
	swig -c++ -python MGSimCore.i
	$(CC) -c -g $(CFLAGS) $(INC) $(LIBS) MGSimCore.cpp MGSimCore_wrap.cxx -I$(PYTHON_INC)
	#$(CC) -shared $(LIBS) $(CFLAGS) $(OBJ) -Wl,-rpath,/usr/local/lib/,-rpath,/usr/local/cuda-10.2/lib64/,-rpath,/usr/lib/x86_64-linux-gnu/ MGSimCore_wrap.o -o _MGSimCore.so -L$(PYTHON_LIB) -Wl,-rpath,"\$ORIGIN"
	$(CC) -shared $(LIBS) $(CFLAGS) $(OBJ) -Wl,-rpath,/usr/local/lib/,-rpath,/usr/lib64/ MGSimCore_wrap.o -o _MGSimCore.so -L$(PYTHON_LIB) -Wl,-rpath,"\$ORIGIN"
$(OBJ_DIR)/%.o: %.cpp
	$(CC) -c -g $(CFLAGS) $(INC) -o $@ $<

$(BIN): $(OUT_DIRS) $(OBJ)
#	$(CC) -g $(CFLAGS) $(OBJ) $(LIBS) -o $@
	$(CC) -g $(CFLAGS) $(INC) $(OBJ) $(LIBS) -o $@

$(OUT_DIRS):
	$(MKDIR) $(OUT_DIRS)
	
clean:
	- rm -f -r $(OUT_DIRS) $(BIN) $(OBJ) $(OBJ_DIR) MGSimCore_wrap.cxx _MGSimCore.so

