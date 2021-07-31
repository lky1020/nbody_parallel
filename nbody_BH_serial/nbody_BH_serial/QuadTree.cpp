#include "QuadTree.h"
#include "Quad.h"
#include <iostream>
#include <cmath>
#include <cassert>

QuadTree::QuadTree() {}

QuadTree::QuadTree(Quad* boundary) :
    boundary_(boundary),
    rect_shape_(sf::Vector2f(boundary->GetWidth(), boundary->GetHeight())),
        rect_(boundary->GetLeft(), boundary->GetTop(), boundary->GetWidth(), boundary->GetHeight()) {

    rect_shape_.setPosition(boundary->GetLeft(), boundary->GetTop());
    rect_shape_.setOutlineThickness(1);
    rect_shape_.setOutlineColor(sf::Color::Green);
    rect_shape_.setFillColor(sf::Color(255, 255, 255, 0));
}

//Calculate Total Mass and Center of Mass (No applicable - Hard to Control)
//void QuadTree::updateMXY(Body* b) {
//    Body *a = this->GetBody();
//
//    double m = a->mass() + b->mass();
//    b->setM(m);
//    b->setX((a->position().x * a->mass() + b->position().x * b->mass()) / m);
//    b->setY((a->position().y * a->mass() + b->position().y * b->mass()) / m);
//}

void QuadTree::addBody(Body* b, float _zoom) {
    Quad nw = GetBoundary()->NW();
    Quad ne = GetBoundary()->NE();
    Quad sw = GetBoundary()->SW();
    Quad se = GetBoundary()->SE();

    if (b->in(nw, _zoom)) {
        NW_->InsertElement(b, _zoom);
    }
    else if (b->in(ne, _zoom)) {
        NE_->InsertElement(b, _zoom);
    }
    else if (b->in(sw, _zoom)) {
        SW_->InsertElement(b, _zoom);
    }
    else if (b->in(se, _zoom)) {
        SE_->InsertElement(b, _zoom);
    }
}

void QuadTree::InsertElement(Body* b, float _zoom) {
    if (body_ == NULL) {
        body_ = b;
        return;
    }

    // internal node
    if (!CheckInternal()) {
        // update the center-of-mass and total mass (Reject)
        //updateMXY(b);

        // recursively insert Body b into the appropriate quadrant
        addBody(b, _zoom);
    }

    else {
        // subdivide the region further by creating four children
        int left = GetRect().left;
        int top = GetRect().top;
        int new_width = GetRect().width / 2;
        int new_height = GetRect().height / 2;

        Quad* NE = new Quad(left + new_width, top, new_width, new_height);
        NE_ = new QuadTree(NE);

        Quad* NW = new Quad(left, top, new_width, new_height);
        NW_ = new QuadTree(NW);

        Quad* SE = new Quad(left + new_width, top + new_height, new_width, new_height);
        SE_ = new QuadTree(SE);

        Quad* SW = new Quad(left, top + new_height, new_width, new_height);
        SW_ = new QuadTree(SW);

        this->divided_ = true;

        // recursively insert both this body and Body b into the appropriate quadrant
        addBody(this->GetBody(), _zoom);
        addBody(b, _zoom);

        // update the center-of-mass and total mass (Reject)
        //updateMXY(b);
    }
}

void QuadTree::UpdateForce(Body* b, float _zoom) {
    if (body_ == NULL || b == body_) {
        return;
    }
    
    // if the current node is external, update net force acting on b
    if (CheckInternal()) {
        b->AddForce(*body_, _zoom);
    }

    // for internal nodes
    else {

        // width of region represented by internal node
        double s = this->GetBoundary()->GetWidth();

        // distance between Body b and this node's center-of-mass
        double d = body_->distanceTo(*b, _zoom);

        // compare ratio (s / d) to threshold value Theta
        if ((s / d) < Theta) {
            b->AddForce(*body_, _zoom);   // b is far away
        }
            
        // recurse on each of current node's children
        else {
            NW_->UpdateForce(b, _zoom);
            NE_->UpdateForce(b, _zoom);
            SW_->UpdateForce(b, _zoom);
            SE_->UpdateForce(b, _zoom);
        }
    }
}

void QuadTree::QtreeFreeMemory() {
    if (GetNW()) {
        GetNW()->QtreeFreeMemory();
        GetNE()->QtreeFreeMemory();
        GetSW()->QtreeFreeMemory();
        GetSE()->QtreeFreeMemory();
    }
    delete this;
}
