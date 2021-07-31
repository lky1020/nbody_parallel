#ifndef Quad_H_
#define Quad_H_

class Quad {
	double _left;
	double _top;
	double _width;
	double _height;

	public:
		Quad(double left, double top, double width, double height) :
			_left(left), _top(top), _width(width), _height(height) {}

		bool contains(double x, double y, float _zoom);

		double GetLeft() const {
			return _left;
		}

		double GetTop() const {
			return _top;
		}

		double GetWidth() const {
			return _width;
		}

		double GetHeight() const {
			return _height;
		}

		Quad NE();
		Quad NW();
		Quad SE();
		Quad SW();
};

#endif // Quad_H_
