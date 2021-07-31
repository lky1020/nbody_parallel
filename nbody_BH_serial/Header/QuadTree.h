#ifndef QuadTree_H_
#define QuadTree_H_

#include "Body.h"
#include <vector>
#include <SFML/Graphics.hpp>

class QuadTree {
public:
    QuadTree(int left, int top, int width, int height);
    bool InsertElement(Body* b);
    void Subdivide(QuadTree* root);
    int GetSize() const {
        return children_.size();
    }
    QuadTree* GetNW() const {
        return NW_;
    }
    QuadTree* GetNE() const {
        return NE_;
    }
    QuadTree* GetSW() const {
        return SW_;
    }
    QuadTree* GetSE() const {
        return SE_;
    }
    sf::IntRect GetRect() const {
        return rect_;
    }
    sf::RectangleShape GetRectShape() const {
        return rect_shape_;
    }
    std::vector<Body*> GetChildren() const {
        return children_;
    }
    void QtreeCheckCollisions(int& num_collisions);
    void QtreeFreeMemory();
private:
    std::vector<Body*> children_;
    QuadTree* NW_ = nullptr;
    QuadTree* NE_ = nullptr;
    QuadTree* SW_ = nullptr;
    QuadTree* SE_ = nullptr;
    sf::RectangleShape rect_shape_;
    sf::IntRect rect_;
};

#endif // QuadTree_H_
