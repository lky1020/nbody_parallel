#include <cmath>
#include <random>

#include "Body.h"
#include "Util.h"

Body& Body::update(double dt) {
  // update velocity
  _velocity.x += _accel.x;
  _velocity.y += _accel.y;
  _velocity.z += _accel.z;

  // reset acceleration
  _accel.x = 0.0;
  _accel.y = 0.0;
  _accel.z = 0.0;

  // update position
  _position.x += TIME_STEP * _velocity.x / TO_METERS;
  _position.y += TIME_STEP * _velocity.y / TO_METERS;
  _position.z += TIME_STEP * _velocity.z / TO_METERS;

  // Another Update calculation (Not Applicable)
  //_velocity.x += _fx / mass();
  //_velocity.y += _fy / mass();
  //_position.x += TIME_STEP * _velocity.x / TO_METERS;
  //_position.y += TIME_STEP * _velocity.y / TO_METERS;

  return *this;
}

bool Body::in(Quad& q, float _zoom) {
    return q.contains(this->position().x, this->position().y, _zoom);
}

//void Body::ResetForce() {
//    this->_fx = 0;
//    this->_fy = 0;
//    this->_fz = 0;
//}

// calculates the effects of an interaction between 2 bodies
void Body::AddForce(Body& b, float _zoom) {
    Body *a = this;

    //Another Force calculation(Not Applicable)
    //double EPS = 0.015 * TO_METERS;      // softening parameter
    //double dx = static_cast<double>(Util::to_pixel_space(b.position().x - a->position().x, WIDTH, _zoom));
    //double dy = static_cast<double>(Util::to_pixel_space(b.position().y - a->position().y, HEIGHT, _zoom));
    //double dist = sqrt(dx * dx + dy * dy);
    //double f = TIME_STEP * (G * a->mass() * b.mass()) / ((dist * dist + EPS * EPS));
    //a->_fx += f * dx / dist;
    //a->_fy += f * dy / dist;

    vec3 posDiff{};  // position difference between the 2 bodies
    posDiff.x = (a->_position.x - b.position().x) * TO_METERS;  // calculate it
    posDiff.y = (a->_position.y - b.position().y) * TO_METERS;
    posDiff.z = (a->_position.z - b.position().z) * TO_METERS;
    // the actual distance is the length of the vector
    auto dist = posDiff.magnitude();
    // calculate force
    double F = TIME_STEP * (G * _mass * b.mass()) / ((dist * dist + SOFTENING * SOFTENING) * dist);

    // set this body's acceleration
    a->_accel.x -= F * posDiff.x / _mass;
    a->_accel.y -= F * posDiff.y / _mass;
    a->_accel.z -= F * posDiff.z / _mass;

    // set the other body's acceleration
    b.acceleration().x += F * posDiff.x / _mass;
    b.acceleration().y += F * posDiff.y / _mass;
    b.acceleration().z += F * posDiff.z / _mass;
}

double Body::distanceTo(Body& b, float _zoom) {
    double dx = static_cast<double>(Util::to_pixel_space(this->position().x - b.position().x, WIDTH, _zoom));
    double dy = static_cast<double>(Util::to_pixel_space(this->position().y - b.position().y, HEIGHT, _zoom));
    return sqrt(dx * dx + dy * dy);
}

// generates n random bodies, stores them in a vector and returns it
std::vector<Body> Body::generate(unsigned int n) {
  // vector to store the results
  std::vector<Body> bodies;
  // make sure there is enough space in it
  bodies.reserve(n);

  // random distributions which will be used to generate a number of values
  using std::uniform_real_distribution;
  uniform_real_distribution<double> randAngle(0.0, 200.0 * PI);  // random angle
  uniform_real_distribution<double> randRadius(INNER_BOUND, SYSTEM_SIZE);  // random radius for the system
  uniform_real_distribution<double> randHeight(0.0, SYSTEM_THICKNESS);  // random z
  uniform_real_distribution<double> randMass(10.0, 100.0);  // random mass
  std::random_device rd;  // make it more random!
  std::mt19937 gen(rd());

  // predeclare some variables
  double angle;
  double radius;
  double velocity;
  // velocity = 0.67*sqrt((G*SOLAR_MASS) / (4 * BINARY_SEPARATION*TO_METERS));
  // sun - put a really heavy body at the centre of the universe
  bodies.emplace_back(vec3{0.0, 0.0, 0.0}, vec3{0.0, 0.0, 0.0},
                      vec3{0.0, 0.0, 0.0}, 50, SOLAR_MASS);

  // extra mass
  double totalExtraMass = 0.0;
  // start at 1 because the sun is at index 0
  for (int index = 1; index < NUM_BODIES; ++index) {
    // generate a random body:
    // get a random angle
    angle = randAngle(gen);
    // get a random radius within the system bounds
    radius = sqrt(SYSTEM_SIZE) * sqrt(randRadius(gen));
    auto t = ((G * (SOLAR_MASS + ((radius - INNER_BOUND) / SYSTEM_SIZE) *
                                     EXTRA_MASS * SOLAR_MASS)));
    velocity = t / (radius * TO_METERS);
    // calculate velocity
    velocity = pow(velocity, 0.5);
    // evenly distributed mass
    auto mass = (EXTRA_MASS * SOLAR_MASS) / NUM_BODIES;
    // keep track of mass
    totalExtraMass += mass;
    // add the body to the vector
    bodies.emplace_back(
        vec3{radius * cos(angle), radius * sin(angle),
             randHeight(gen) - SYSTEM_THICKNESS / 2},              // position
        vec3{velocity * sin(angle), -velocity * cos(angle), 0.0},  // velocity
        vec3{0.0, 0.0, 0.0},  // acceleration
        radius,               // radius
        mass);                // mass
  }

  // return result
  return bodies;
}

/* getters and setters */
const vec3& Body::position() const { return _position; }

const vec3& Body::velocity() const { return _velocity; }

vec3& Body::acceleration() { return _accel; }

double Body::radius() const { return _radius; }

double Body::mass() const { return _mass; }
