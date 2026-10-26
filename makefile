CXX = g++
AR = ar
ARFLAGS = rcs
LDDIR = -Llib
DEFAULT_LDFLAGS = -lmatrix
LDFLAGS = 
DEFAULT_CXXFLAGS = -std=c++17 -pedantic -Wfatal-errors -Wconversion -Wredundant-decls -Wshadow -Wall -Wextra
CXXFLAGS = 
INCLUDE_DIR = -Iinclude
BINFLAGS =

CXXLIB = lib/libjipks.a
LUALIB = lua/jipks.so
APP = bin/jipks
SRCDIR = src
SRC = $(shell find $(SRCDIR) -name "*.cpp")
OBJDIR = obj
OBJ = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRC))

# On exclut main.o lors de la liaison des tests
OBJ_NO_MAIN = $(filter-out $(OBJDIR)/main.o, $(OBJ))
OBJ_LUA_LIB = $(filter-out $(OBJDIR)/main.o, $(OBJ))
OBJ_WITHOUT_LUA = $(filter-out $(OBJDIR)/lua.o, $(OBJ))
OBJ_CXX_LIB = $(filter-out $(OBJDIR)/main.o $(OBJDIR)/lua.o, $(OBJ))

.PHONY: all run clean debug doc init lib test release

# Compilation du binaire simple
all: CXXFLAGS := $(DEFAULT_CXXFLAGS)
all: LDFLAGS := $(DEFAULT_LDFLAGS)
all: $(APP)

$(APP): $(OBJ_WITHOUT_LUA)
	@mkdir -p bin
	$(CXX) $(LDDIR) -o $(APP) $^ $(BINFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $(INCLUDE_DIR) $< -o $@

# Compilation du binaire en tant que librairie.
$(CXXLIB): $(OBJ_CXX_LIB)
	@mkdir -p lib
	@rm -rR include
	@mkdir -p include
	cp -rR $(SRCDIR)/matrix/ include
	find include -type f -name "*.cpp" -delete
	$(AR) $(ARFLAGS) $@ $^

$(LUALIB): $(OBJ_LUA_LIB)
	@mkdir -p lua
	$(CXX) $(LDDIR) -o $@ $^ $(LDFLAGS)

run: 
	$(APP)

lib: CXXFLAGS := $(DEFAULT_CXXFLAGS) -DNDEBUG
lib: LDFLAGS := $(DEFAULT_LDFLAGS)
lib: clean $(CXXLIB)

lua: INCLUDE_DIR := -I/usr/include/lua5.4
lua: LDFLAGS := $(DEFAULT_LDFLAGS) -shared -llua5.4
lua: CXXFLAGS := $(DEFAULT_CXXFLAGS) -DNDEBUG -fPIC
lua: clean $(LUALIB)

clean:
	find $(OBJDIR) -type f -name "*.o" -delete

debug: CXXFLAGS += -g -DDEBUG
debug: clean $(APP)
	gdb $(APP)

doc:
	doxygen Doxyfile

init:
	mkdir -p bin
	mkdir -p obj
