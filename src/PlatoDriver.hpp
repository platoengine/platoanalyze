#ifndef PLATO_DRIVER_HPP
#define PLATO_DRIVER_HPP

#include <string>
#include <vector>
#include <memory>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Omega_h_tag.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_teuchos.hpp>

#include "AnalyzeOutput.hpp"
#include "PlatoUtilities.hpp"
#include "PlatoProblemFactory.hpp"
//#include "StructuralDynamicsOutput.hpp"

namespace Plato
{


template<const Plato::OrdinalType SpatialDim>
void run(Teuchos::ParameterList& aProblemSpec,
         Omega_h::Mesh& aMesh,
         Omega_h::MeshSets& aMeshSets,
         const std::string & aVizFilePath)
{
    // create mesh based density from host data
    std::vector<Plato::Scalar> tControlHost(aMesh.nverts(), 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tControlHostView(tControlHost.data(), tControlHost.size());
    auto tControl = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tControlHostView);

    // Solve Plato problem
    Plato::ProblemFactory<SpatialDim> tProblemFactory;
    std::shared_ptr<::Plato::AbstractProblem> tPlatoProblem = tProblemFactory.create(aMesh, aMeshSets, aProblemSpec);
    auto tSolution     = tPlatoProblem->solution(tControl);
    auto tStateDataMap = tPlatoProblem->getDataMap();

    Plato::output<SpatialDim>(aProblemSpec, aVizFilePath, tSolution, tStateDataMap, aMesh);
}

template<const Plato::OrdinalType SpatialDim>
void driver(Omega_h::Library* aLibOSH,
            Teuchos::ParameterList & aProblemSpec,
            const std::string& aInputFilename,
            const std::string& aVizFilePath)
{
    Omega_h::Mesh tMesh = Omega_h::read_mesh_file(aInputFilename, aLibOSH->world());
    tMesh.set_parting(Omega_h_Parting::OMEGA_H_GHOSTED);

    Omega_h::Assoc tAssoc;
    if (aProblemSpec.isSublist("Associations"))
    {
      auto& tAssocParamList = aProblemSpec.sublist("Associations");
      Omega_h::update_assoc(&tAssoc, tAssocParamList);
    } 
    else {
      tAssoc[Omega_h::NODE_SET] = tMesh.class_sets;
      tAssoc[Omega_h::SIDE_SET] = tMesh.class_sets;
    }
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&tMesh, tAssoc);
    
    Plato::run<SpatialDim>(aProblemSpec, tMesh, tMeshSets, aVizFilePath);
}

void driver(Omega_h::Library* aLibOmegaH,
            Teuchos::ParameterList & aProblemSpec,
            const std::string& aInputFilename,
            const std::string& aVizFilePath)
{
    const Plato::OrdinalType tSpaceDim = aProblemSpec.get<Plato::OrdinalType>("Spatial Dimension", 3);

    // Run Plato problem
    if(tSpaceDim == static_cast<Plato::OrdinalType>(3))
    {
        #ifdef PLATOANALYZE_3D
        driver<3>(aLibOmegaH, aProblemSpec, aInputFilename, aVizFilePath);
        #else
        throw std::runtime_error("3D physics is not compiled.");
        #endif
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(2))
    {
        #ifdef PLATOANALYZE_2D
        driver<2>(aLibOmegaH, aProblemSpec, aInputFilename, aVizFilePath);
        #else
        throw std::runtime_error("2D physics is not compiled.");
        #endif
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(1))
    {
        #ifdef PLATOANALYZE_1D
        driver<1>(aLibOmegaH, aProblemSpec, aInputFilename, aVizFilePath);
        #else
        throw std::runtime_error("1D physics is not compiled.");
        #endif
    }
}

} // namespace Plato

#endif /* #ifndef PLATO_DRIVER_HPP */

