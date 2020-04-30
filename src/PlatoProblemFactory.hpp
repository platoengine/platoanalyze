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

#include "elliptic/Problem.hpp"
#include "EllipticVMSProblem.hpp"
#include "parabolic/Problem.hpp"
#include "AnalyzeMacros.hpp"

#include "Mechanics.hpp"
#include "PlasticityProblem.hpp"
#include "StabilizedMechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"
#include "StabilizedThermomechanics.hpp"
#include "hyperbolic/HyperbolicProblem.hpp"
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
            if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Mechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tInputData);
                return tOutput;
            }
            else 
            if(tPDE == "Hyperbolic")
            {
                return std::make_shared < HyperbolicProblem<::Plato::Hyperbolic::Mechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
            }
            else
            if(tPDE == "Infinite Strain Plasticity")
            {
                auto tOutput = std::make_shared < PlasticityProblem<::Plato::InfinitesimalStrainPlasticity<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tInputData);
                return tOutput;
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown PDE type (" << tPDE << ") requested.";
                THROWERR(ss.str());
            }
        }
        else if(tPhysics == "Stabilized Mechanical")
        {
            if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < EllipticVMSProblem<::Plato::StabilizedMechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tInputData);
                return tOutput;
            }
            else
            {
                std::stringstream tStringStream;
                tStringStream << "Unknown PDE type (" << tPDE << ") requested.";
                THROWERR(tStringStream.str());
            }
        }
        else if(tPhysics == "Thermal")
        {
            if(tPDE == "Parabolic")
            {
                return std::make_shared < Plato::Parabolic::Problem<::Plato::Thermal<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
            }
            else if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Thermal<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tInputData);
                return tOutput;
            }
            else
            {
                std::stringstream tStringStream;
                tStringStream << "Unknown PDE type (" << tPDE << ") requested.";
                THROWERR(tStringStream.str());
            }
        }
        else if(tPhysics == "StructuralDynamics")
        {
//            return std::make_shared<Plato::StructuralDynamicsProblem<Plato::StructuralDynamics<SpatialDim>>>(aMesh, aMeshSets, tInputData);
        }
        else if(tPhysics == "Electromechanical")
        {
            auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Electromechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
            tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tInputData);
            return tOutput;
        }
        else if(tPhysics == "Stabilized Thermomechanical")
        {
            if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < EllipticVMSProblem<::Plato::StabilizedThermomechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tInputData);
                return tOutput;
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown PDE type (" << tPDE << ") requested.";
                THROWERR(ss.str());
            }
        }
        else if(tPhysics == "Thermomechanical")
        {
            if(tPDE == "Parabolic")
            {
                return std::make_shared < Plato::Parabolic::Problem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
            }
            else if(tPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < Plato::Elliptic::Problem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aMeshSets, tInputData, aMachine);
                tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tInputData);
                return tOutput;
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown PDE type (" << tPDE << ") requested.";
                THROWERR(ss.str());
            }
        }
        else
        {
            std::stringstream tStringStream;
            tStringStream << "Unknown Physics type (" << tPhysics << ") requested.";
            THROWERR(tStringStream.str());
        }

        return (nullptr);
    }
};
// class ProblemFactory

}// namespace Plato

#endif /* PLATOPROBLEMFACTORY_HPP_ */
