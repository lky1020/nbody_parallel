#include "Simulation.h"
#include "Timer.h"
#include "Util.h"
#include "QuadTree.h"
#include "Quad.h"

#include <omp.h>
#include <random>
#include <cmath>

// Global Variable
QuadTree* qtree = new QuadTree();
bool onDebug = false;

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
    if (event.type == sf::Event::MouseWheelScrolled)  // check if mouse scroll is used
    {
      // zoom in/out accordingly
      _zoom *= 1.f + (-event.mouseWheelScroll.delta / 10.f);
      _view.zoom(1.f + (-event.mouseWheelScroll.delta / 10.f));
    }
    if (event.type == sf::Event::KeyPressed)
    {
        if (event.key.code == sf::Keyboard::A) {
            _view.move(20.0f, 0.0f);
        }
        if (event.key.code == sf::Keyboard::D) {
            _view.move(-20.0f, 0.0f);
        }
        if (event.key.code == sf::Keyboard::W) {
            _view.move(0.0f, 20.0f);
        }
        if (event.key.code == sf::Keyboard::S) {
            _view.move(0.0f, -20.0f);
        }
        if (event.key.code == sf::Keyboard::Space)
        {
            onDebug = !onDebug;
        }
    }
  }

  // don't forget to set the view after modifying it
  _window.setView(_view);
}

void DrawQTree(QuadTree* root, sf::RenderWindow& window) {
    if (onDebug) {
        QuadTree* curr = root;
        if (curr->GetDivided()) {
            DrawQTree(curr->GetNW(), window);
            DrawQTree(curr->GetNE(), window);
            DrawQTree(curr->GetSW(), window);
            DrawQTree(curr->GetSE(), window);
        }
        else {
            window.draw(curr->GetRectShape());
        }
    }
}

// updates all bodies
void Simulation::update() {
  Timer t(__func__);
  Quad* boundary = new Quad(0, 0, _width, _height);
  qtree = new QuadTree(boundary);

  //Add body into quadTree
  for (int i = 0; i < NUM_BODIES - 1; ++i) {
    qtree->InsertElement(&_bodies[i], _zoom);
  }

  // update the forces, positions, velocities, and accelerations
#pragma omp parallel for num_threads(8) //speed up the update process with openMP
  for (int i = 0; i < NUM_BODIES; i++) {
      //_bodies[i].ResetForce(); //_fx, _fy, _fz not used
      qtree->UpdateForce(&_bodies[i], _zoom);

      // Prevent sun to move
      if (i != 0) {
          _bodies[i].update();
      }
  }
}

// renders all bodies
void Simulation::render() {
  // clear screen
  _window.clear(sf::Color::Black);

  DrawQTree(qtree, _window);

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
  qtree->QtreeFreeMemory();
}
