CXX = g++
CXXFLAGS = -std=c++11
SRC = subtask1.cpp
TARGET = subtask1

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
