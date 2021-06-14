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

/***************************************************************************//**
 * \brief Run simulation with Plato Analyze.
 *
 * \tparam SpatialDim spatial dimensions
 *
 * \param [in] aInputData   input parameters list
 * \param [in] aMesh        mesh database
 * \param [in] aMeshSets    side sets database
 * \param [in] aVizFilePath output visualization file path
*******************************************************************************/
template<const Plato::OrdinalType SpatialDim>
void run(Teuchos::ParameterList& aInputData,
         Comm::Machine           aMachine,
         Omega_h::Mesh&          aMesh,
         Omega_h::MeshSets&      aMeshSets)
{
    // create mesh based density from host data
    std::vector<Plato::Scalar> tControlHost(aMesh.nverts(), 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tControlHostView(tControlHost.data(), tControlHost.size());
    auto tControl = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tControlHostView);

    // Solve Plato problem
    Plato::ProblemFactory<SpatialDim> tProblemFactory;
    std::shared_ptr<::Plato::AbstractProblem> tPlatoProblem = tProblemFactory.create(aMesh, aMeshSets, aInputData, aMachine);
    auto tSolution = tPlatoProblem->solution(tControl);
    if(false){ tSolution.print(); }

    auto tPlatoProblemList = aInputData.sublist("Plato Problem");
    if (tPlatoProblemList.isSublist("Criteria"))
    {
        auto tCriteriaList = tPlatoProblemList.sublist("Criteria");
        for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaList.begin(); tIndex != tCriteriaList.end(); ++tIndex)
        {
            std::string tName = tCriteriaList.name(tIndex);
            Plato::Scalar tCriterionValue = tPlatoProblem->criterionValue(tControl, tSolution, tName);
            printf("Criterion '%s' , Value %0.10e\n", tName.c_str(), tCriterionValue);
        }
    }

    auto tFilepath = aInputData.get<std::string>("Output Viz");
    tPlatoProblem->output(tFilepath);
}
// function run


/***************************************************************************//**
 * \brief Run simulation with Plato Analyze.
 *
 * \tparam SpatialDim spatial dimensions
 *
 * \param [in] aLibOSH      pointer to Omega_h library database
 * \param [in] aInputData   input parameters list
 * \param [in] aInputFile   Plato Analyze input file name
 * \param [in] aVizFilePath output visualization file path
*******************************************************************************/
template<const Plato::OrdinalType SpatialDim>
void driver(Omega_h::Library*        aLibOSH,
            Teuchos::ParameterList & aInputData,
            Comm::Machine            aMachine)
{
    auto tInputMesh = aInputData.get<std::string>("Input Mesh");

    Omega_h::Mesh tMesh = Omega_h::read_mesh_file(tInputMesh, aLibOSH->world());
    tMesh.set_parting(Omega_h_Parting::OMEGA_H_GHOSTED);

    Omega_h::Assoc tAssoc;
    if (aInputData.isSublist("Associations"))
    {
      auto& tAssocParamList = aInputData.sublist("Associations");
      Omega_h::update_assoc(&tAssoc, tAssocParamList);
    } 
    else {
      tAssoc[Omega_h::ELEM_SET] = tMesh.class_sets;
      tAssoc[Omega_h::NODE_SET] = tMesh.class_sets;
      tAssoc[Omega_h::SIDE_SET] = tMesh.class_sets;
    }
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&tMesh, tAssoc);
    
    Plato::run<SpatialDim>(aInputData, aMachine, tMesh, tMeshSets);
}
// function driver

/***************************************************************************//**
 * \brief Run simulation with Plato Analyze.
 *
 * \param [in] aLibOSH      pointer to Omega_h library database
 * \param [in] aInputData   input parameters list
 * \param [in] aInputFile   Plato Analyze input file name
 * \param [in] aVizFilePath output visualization file path
*******************************************************************************/
void driver(Omega_h::Library* aLibOmegaH,
            Teuchos::ParameterList & aInputData,
            Comm::Machine            aMachine)
{
    const Plato::OrdinalType tSpaceDim = aInputData.get<Plato::OrdinalType>("Spatial Dimension", 3);

    // Run Plato problem
    if(tSpaceDim == static_cast<Plato::OrdinalType>(3))
    {
        #ifdef PLATOANALYZE_3D
        driver<3>(aLibOmegaH, aInputData, aMachine);
        #else
        throw std::runtime_error("3D physics option is not compiled.");
        #endif
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(2))
    {
        #ifdef PLATOANALYZE_2D
        driver<2>(aLibOmegaH, aInputData, aMachine);
        #else
        throw std::runtime_error("2D physics option is not compiled.");
        #endif
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(1))
    {
        #ifdef PLATOANALYZE_1D
        driver<1>(aLibOmegaH, aInputData, aMachine);
        #else
        throw std::runtime_error("1D physics option is not compiled.");
        #endif
    }
}
// function driver


}
// namespace Plato

#endif /* #ifndef PLATO_DRIVER_HPP */

