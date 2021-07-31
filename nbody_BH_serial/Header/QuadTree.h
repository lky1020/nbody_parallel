#ifndef QuadTree_H_
#define QuadTree_H_

#include "Body.h"
#include "Quad.h";
#include <vector>
#include <SFML/Graphics.hpp>

class QuadTree {
public:
    QuadTree();
    QuadTree(Quad* boundary);
    void InsertElement(Body* b, float _zoom);
    //void updateMXY(Body* b);
    void addBody(Body* b, float _zoom);

    //Calculate Force
    void UpdateForce(Body* b, float _zoom);

    bool CheckInternal() const {
        return (GetNW() == NULL && GetNE() == NULL && GetSW() == NULL && GetSE() == NULL);
    }

    bool CheckExternal(QuadTree t) const {
        if (t.GetNW() == NULL && t.GetNE() == NULL && t.GetSW() == NULL && t.GetSE() == NULL)
            return true;

        return false;
    }

    Body* GetBody() const {
        return body_;
    }
    Quad* GetBoundary() const {
        return boundary_;
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
    bool GetDivided() const {
        return divided_;
    }
    sf::IntRect GetRect() const {
        return rect_;
    }
    sf::RectangleShape GetRectShape() const {
        return rect_shape_;
    }
    //void QtreeCheckCollisions(int& num_collisions);
    void QtreeFreeMemory();
private:
    const double Theta = 0.5;
    Body* body_;
    Quad* boundary_;
    QuadTree* NW_ = nullptr;
    QuadTree* NE_ = nullptr;
    QuadTree* SW_ = nullptr;
    QuadTree* SE_ = nullptr;
    bool divided_ = false;
    sf::RectangleShape rect_shape_;
    sf::IntRect rect_;
};

#endif // QuadTree_H_
