#include <iostream>

#include "palabos2D.h"
#include "palabos2D.hh"
#include "palabos3D.h"
#include "palabos3D.hh"
#include <chrono>
#include <mpi.h>

using namespace plb;

template <typename T>
class SphereShape3D : public DomainFunctional3D {
private:
    plint cx, cy, cz;
    plint radiusSqr;
public:
    SphereShape3D(plint cx_, plint cy_, plint cz_, plint radius)
            : cx(cx_), cy(cy_), cz(cz_), radiusSqr(radius * radius) {}

private:
    bool operator()(plint iX, plint iY, plint iZ) const override {
        return util::sqr(cx - iX) + util::sqr(cy - iY) + util::sqr(cz - iZ) < radiusSqr;
    }

    [[nodiscard]] DomainFunctional3D *clone() const override {
        return new SphereShape3D<T>(*this);
    }

};

void createLattice(long dimSize) {
    MultiBlockLattice3D<float, descriptors::D3Q19Descriptor>
            lattice(dimSize, dimSize, dimSize, new BGKdynamics<float, descriptors::D3Q19Descriptor>(0.65f));

    OnLatticeBoundaryCondition3D<float, descriptors::D3Q19Descriptor>* boundaryCondition =
            createLocalBoundaryCondition3D<float, descriptors::D3Q19Descriptor>();

    boundaryCondition->setVelocityConditionOnBlockBoundaries(lattice, boundary::dirichlet);
    setBoundaryVelocity(lattice, lattice.getBoundingBox(), Array<float, 3>(0.1f, 0, 0));

    defineDynamics(lattice, lattice.getBoundingBox(),
                   new SphereShape3D<float>(50, 50, 8, 15),
                   new BounceBack<float, descriptors::D3Q19Descriptor>);

    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), 1.f, Array<float, 3>(0.1f, 0, 0));

    lattice.initialize();

    lattice.toggleInternalStatistics(false);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; i++) {
        lattice.collideAndStream();
    }
    int proc_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    if (proc_id == 0) {
        std::chrono::duration<float> d = std::chrono::steady_clock::now() - start;
        std::cout << d.count() << std::endl;
    }
}


int main(int argc, char** argv) {
    plbInit(&argc, &argv);

    long dim = 100;
    if (argc == 2) {
        dim = atoi(argv[1]);
    }

    createLattice(dim);

}
