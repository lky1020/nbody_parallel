#ifndef BODY_H
#define BODY_H

#include "Constants.h"
#include "Util.h"
#include "Quad.h"

#include <iostream>
#include <vector>

/*
 * This class describes a single body in the simulation.
 */

class Body {
  vec3 _position;  // position
  vec3 _velocity;  // velocity
  vec3 _accel;     // acceleration
  double _radius;
  double _mass;    // mass

 public:
  //double _fx, _fy, _fz;   // force (Reject)

  Body(vec3 pos, vec3 vel, vec3 accel, double radius, double mass)
      : _position(pos), _velocity(vel), _accel(accel), _radius(radius), _mass(mass) {}

  Body& update(double dt);
  bool in(Quad& q, float _zoom);
  //void ResetForce();      // used by _fx, _fy, _fz
  void AddForce(Body& b, float _zoom);
  double distanceTo(Body& b, float _zoom);

  static std::vector<Body> generate(unsigned int);

  const vec3& position() const;
  const vec3& velocity() const;
  vec3& acceleration();  // not const so that the acceleration can be modified;
                         // probably should add a setter instead
  double radius() const;
  double mass() const;

  void setM(double newM) {
      this->_mass = newM;
  }

  void setX(double newX) {
      this->_position.x = newX;
  }

  void setY(double newY) {
      this->_position.y = newY;
  }

  // will be used for debugging
  friend std::ostream& operator<<(std::ostream& str, const Body& p) {
    return str << "Position: " << p.position() << "; Velocity: "
               << p.velocity()
               //<< "; Acceleration: " << p.acceleration()
               << "; Mass: " << p.mass() << std::endl;
  }
};

#endif /* BODY_H */
