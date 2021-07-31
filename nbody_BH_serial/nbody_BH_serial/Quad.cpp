#include "Quad.h"
#include "Util.h"
#include "Constants.h"

bool Quad::contains(double x, double y, float _zoom) {
    x = static_cast<double>(Util::to_pixel_space(x, WIDTH, _zoom));
    y = static_cast<double>(Util::to_pixel_space(y, HEIGHT, _zoom));

    return (x >= this->_left - this->_width &&
        x <= this->_left + this->_width &&
        y >= this->_top - this->_height &&
        y <= this->_top + this->_height);
}

Quad Quad::NE() {
    return Quad(this->_left + this->_width / 2, this->_top, this->_width / 2, this->_height / 2);
}

Quad Quad::NW() {
    return Quad(this->_left, this->_top, this->_width / 2, this->_height / 2);
}

Quad Quad::SE() {
    return Quad(this->_left + this->_width / 2, this->_top + this->_height / 2, this->_width / 2, this->_height / 2);
}

Quad Quad::SW() {
    return Quad(this->_left, this->_top + this->_height / 2, this->_width / 2, this->_height / 2);
}