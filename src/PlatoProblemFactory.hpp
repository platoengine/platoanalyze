/*
 * PlatoProblemFactory.hpp
 *
 *  Created on: Apr 19, 2018
 */

#ifndef PLATOPROBLEMFACTORY_HPP_
#define PLATOPROBLEMFACTORY_HPP_

#include <memory>
#include <sstream>
#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_PLASTICITY
#include "PlasticityProblem.hpp"
#endif

#ifdef PLATO_ELLIPTIC
#include "elliptic/Problem.hpp"
#include "elliptic/updated_lagrangian/Problem.hpp"
#endif

#ifdef PLATO_PARABOLIC
#include "parabolic/Problem.hpp"
#endif

#ifdef PLATO_HYPERBOLIC
#include "hyperbolic/HyperbolicProblem.hpp"
#endif

#ifdef PLATO_STABILIZED
#include "EllipticVMSProblem.hpp"
#include "StabilizedMechanics.hpp"
#include "StabilizedThermomechanics.hpp"
#endif

//#include "StructuralDynamicsProblem.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Manages the construction of a PLATO problem, i.e. physics simulation, in
 * PLATO Analyze.  There are multiple options: 1) mechanics, thermal, thermo-mechanics,
 * and electro-statics.  Also, PLATO Analyze supports transient simulations for thermal
 * and thermo-mechanics problems.
 **********************************************************************************/
template<int SpatialDim>
class ProblemFactory
{
public:
    /******************************************************************************//**
     * \brief Returns a shared pointer to a PLATO problem
     * \param [in] aMesh mesh metadata
     * \param [in] aMeshSets sidesets mesh metadata
     * \param [in] aInputParams xml metadata
     * \returns shared pointer to a PLATO problem
     **********************************************************************************/
    std::shared_ptr<Plato::AbstractProblem> create(Omega_h::Mesh&          aMesh,
                                                   Omega_h::MeshSets&      aMeshSets,
                                                   Teuchos::ParameterList& aInputParams,
                                                   Comm::Machine           aMachine)
    {

        auto tInputData = aInputParams.sublist("Plato Problem");
        auto tPhysics = tInputData.get < std::string > ("Physics");
        auto tPDE = tInputData.get < std::string > ("PDE Constraint");

        if(tPhysics == "Mechanical")
        {
#ifdef PLATO_ELLIPTIC
            if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Mechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(tInputData);
                return tOutput;
            }
            else 
            if(tPDE == "Updated Lagrangian Elliptic")
            {
                using PhysicsType = Plato::Elliptic::UpdatedLagrangian::Mechanics<SpatialDim>;
                auto tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType>> (aMesh, aMeshSets, tInputData, aMachine);
                return tOutput;
            }
#endif
#ifdef PLATO_HYPERBOLIC
            else
            if(tPDE == "Hyperbolic")
            {
                return std::make_shared < HyperbolicProblem<::Plato::Hyperbolic::Mechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
            }
            else
            {
                THROWERR(std::string("Requested 'PDE' keyword '") + tPDE + "' is not supported.");
            }
#endif
#ifdef PLATO_PLASTICITY
        }
        else if(tPhysics == "Plasticity")
        {
            if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < PlasticityProblem<::Plato::InfinitesimalStrainPlasticity<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(tInputData);
                return tOutput;
            }
            else
            {
                THROWERR(std::string("Requested 'PDE' keyword '") + tPDE + "' is not supported.");
            }
        }
#endif
#ifdef PLATO_STABILIZED
        else if(tPhysics == "Stabilized Mechanical")
        {
  #ifdef PLATO_ELLIPTIC
            if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < EllipticVMSProblem<::Plato::StabilizedMechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(tInputData);
                return tOutput;
            }
            else
  #endif
            {
                THROWERR(std::string("Requested 'PDE' keyword '") + tPDE + "' is not supported.");
            }
        }
#endif
        else if(tPhysics == "Thermal")
        {
#ifdef PLATO_PARABOLIC
            if(tPDE == "Parabolic")
            {
                return std::make_shared < Plato::Parabolic::Problem<::Plato::Thermal<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
            }
            else
#endif
#ifdef PLATO_ELLIPTIC
            if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Thermal<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(tInputData);
                return tOutput;
            }
            else
#endif
            {
                THROWERR(std::string("Requested 'PDE' keyword '") + tPDE + "' is not supported.");
            }
        }
        else
        if(tPhysics == "StructuralDynamics")
        {
//            return std::make_shared<Plato::StructuralDynamicsProblem<Plato::StructuralDynamics<SpatialDim>>>(aMesh, aMeshSets, tInputData);
        }
        else
        if(tPhysics == "Electromechanical")
        {
            auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Electromechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
            tOutput->readEssentialBoundaryConditions(tInputData);
            return tOutput;
        }
#ifdef PLATO_STABILIZED
        else
        if(tPhysics == "Stabilized Thermomechanical")
        {
            if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < EllipticVMSProblem<::Plato::StabilizedThermomechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(tInputData);
                return tOutput;
            }
            else
            {
                THROWERR(std::string("Requested 'PDE' keyword '") + tPDE + "' is not supported.");
            }
        }
#endif
        else
        if(tPhysics == "Thermomechanical")
        {
#ifdef PLATO_PARABOLIC
            if(tPDE == "Parabolic")
            {
                return std::make_shared < Plato::Parabolic::Problem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
            }
            else
#endif
#ifdef PLATO_ELLIPTIC
            if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(tInputData);
                return tOutput;
            }
            else
#endif
            {
                THROWERR(std::string("Requested 'PDE' keyword '") + tPDE + "' is not supported.");
            }
        }
        else
        {
            THROWERR(std::string("Requested 'Physics' keyword '") + tPhysics + "' is not supported.");
        }
        return nullptr;
    }
};
// class ProblemFactory

}// namespace Plato

#endif /* PLATOPROBLEMFACTORY_HPP_ */
