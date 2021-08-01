#include "Simulation.h"
#include "Timer.h"
#include "Util.h"

#include <random>

Simulation::Simulation(unsigned int width, unsigned int height)
    : _width(width),
      _height(height),
      _bodies(Body::generate(NUM_BODIES)),
      _zoom(1.2f) {
  // create the window
  _window.create(sf::VideoMode(_width, _height), "N-body simulation",
                 sf::Style::Default);
  // setup the view
  _view.reset(sf::FloatRect(0, 0, _width, _height));
  _view.zoom(_zoom);
  _view.setViewport(sf::FloatRect(0.f, 0.f, 1.f, 1.f));
  // use it
  _window.setView(_view);
}

// starts the simulation
void Simulation::start() {
  // while the window is open
  while (_window.isOpen()) {
    poll_events();  // check for sfml events
    update();       // update bodies
    render();       // render bodies
  }

  std::cout << "Done!" << std::endl;
}

// check for sfml events
void Simulation::poll_events() {
  sf::Event event{};

  // poll events
  while (_window.pollEvent(event)) {
    if (event.type == sf::Event::Closed) {  // check if window is closed
      _window.close();
    }
    if (event.type ==
        sf::Event::MouseWheelScrolled)  // check if mouse scroll is used
    {
      // zoom in/out accordingly
      _zoom *= 1.f + (-event.mouseWheelScroll.delta / 10.f);
      _view.zoom(1.f + (-event.mouseWheelScroll.delta / 10.f));
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) {
        _view.move(20.0f, 0.0f);
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) {
        _view.move(-20.0f, 0.0f);
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) {
        _view.move(0.0f, 20.0f);
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) {
        _view.move(0.0f, -20.0f);
    }
  }

  // don't forget to set the view after modifying it
  _window.setView(_view);
}

// updates all bodies
void Simulation::update() {
  Timer t(__func__);

  // every body interacts with every other body
  // which is why the nested loops are needed
  for (int i = 0; i < NUM_BODIES - 1; ++i) {
    for (int j = i + 1; j < NUM_BODIES; ++j) {
      _bodies[i].interact(_bodies[j]);
    }
  }

  // update bodies' positions
  for (int i = 0; i < NUM_BODIES; ++i) {
    _bodies[i].update();
  }
}

// renders all bodies
void Simulation::render() {
  // clear screen
  _window.clear(sf::Color::Black);

  // temporary circle shape
  sf::CircleShape star(DOT_SIZE, 50);
  star.setOrigin(sf::Vector2f(DOT_SIZE / 2.0f, DOT_SIZE / 2.0f));

  for (size_t i = 0; i < NUM_BODIES; ++i) {
    // get a reference to the current body
    auto current = &_bodies[i];
    // get position and velocity
    auto pos = current->position();
    auto vel = current->velocity();
    // calculate magnitude of the velocity vector
    auto mag = vel.magnitude();

    // orthogonal projection
    // the simulation is essentially in 2D because the z coordinates are not
    // really used so this doesn't really do anything
    auto x = static_cast<double>(Util::to_pixel_space(pos.x, WIDTH, _zoom));
    auto y = static_cast<double>(Util::to_pixel_space(pos.y, HEIGHT, _zoom));
    // calculate a suitable colour
    star.setFillColor(Util::get_dot_colour(x, y, mag));
    // set the position of the circle shape
    star.setPosition(sf::Vector2f(x, y));
    star.setScale(sf::Vector2f(PARTICLE_MAX_SIZE, PARTICLE_MAX_SIZE));
    // the sun is stored at index 0
    if (i == 0) {
      // make the sun bigger and red
      star.setScale(sf::Vector2f(2.5f, 2.5f));
      star.setFillColor(sf::Color::Red);
    }
    // render
    _window.draw(star);
  }

  // display
  _window.display();
}
