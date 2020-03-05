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

#include "EllipticProblem.hpp"
#include "EllipticVMSProblem.hpp"
#include "ParabolicProblem.hpp"
#include "AnalyzeMacros.hpp"

#include "Mechanics.hpp"
#include "StabilizedMechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"
#include "StabilizedThermomechanics.hpp"
#include "hyperbolic/HyperbolicProblem.hpp"
//#include "StructuralDynamics.hpp"
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
    std::shared_ptr<Plato::AbstractProblem> create(Omega_h::Mesh& aMesh,
                                                   Omega_h::MeshSets& aMeshSets,
                                                   Teuchos::ParameterList& aInputParams)
    {

        auto tProblemSpecs = aInputParams.sublist("Plato Problem");
        auto tProblemPhysics = tProblemSpecs.get < std::string > ("Physics");
        auto tProblemPDE = tProblemSpecs.get < std::string > ("PDE Constraint");

        if(tProblemPhysics == "Mechanical")
        {
            if(tProblemPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < EllipticProblem<::Plato::Mechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
                tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tProblemSpecs);
                return tOutput;
            }
            if(tProblemPDE == "Hyperbolic")
            {
                return std::make_shared < HyperbolicProblem<::Plato::Hyperbolic::Mechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(ss.str());
            }
        }
        else if(tProblemPhysics == "Stabilized Mechanical")
        {
            if(tProblemPDE == "Elliptic")
            {
                return std::make_shared < EllipticVMSProblem<::Plato::StabilizedMechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else
            {
                std::stringstream tStringStream;
                tStringStream << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(tStringStream.str());
            }
        }
        else if(tProblemPhysics == "Thermal")
        {
            if(tProblemPDE == "Heat Equation")
            {
                return std::make_shared < ParabolicProblem<::Plato::Thermal<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else if(tProblemPDE == "Thermostatics")
            {
                auto tOutput = std::make_shared < EllipticProblem<::Plato::Thermal<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
                tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tProblemSpecs);
                return tOutput;
            }
            else
            {
                std::stringstream tStringStream;
                tStringStream << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(tStringStream.str());
            }
        }
        else if(tProblemPhysics == "StructuralDynamics")
        {
//            return std::make_shared<Plato::StructuralDynamicsProblem<Plato::StructuralDynamics<SpatialDim>>>(aMesh, aMeshSets, tProblemSpecs);
        }
        else if(tProblemPhysics == "Electromechanical")
        {
            auto tOutput = std::make_shared < EllipticProblem<::Plato::Electromechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tProblemSpecs);
            return tOutput;
        }
        else if(tProblemPhysics == "Stabilized Thermomechanical")
        {
            if(tProblemPDE == "Elliptic")
            {
                return std::make_shared < EllipticVMSProblem<::Plato::StabilizedThermomechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(ss.str());
            }
        }
        else if(tProblemPhysics == "Thermomechanical")
        {
            if(tProblemPDE == "Parabolic")
            {
                return std::make_shared < ParabolicProblem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
            }
            else if(tProblemPDE == "Elliptic")
            {
                auto tOutput = std::make_shared < EllipticProblem<::Plato::Thermomechanics<SpatialDim>> > (aMesh, aMeshSets, tProblemSpecs);
                tOutput->readEssentialBoundaryConditions(aMesh, aMeshSets, tProblemSpecs);
                return tOutput;
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown PDE type (" << tProblemPDE << ") requested.";
                THROWERR(ss.str());
            }
        }
        else
        {
            std::stringstream tStringStream;
            tStringStream << "Unknown Physics type (" << tProblemPhysics << ") requested.";
            THROWERR(tStringStream.str());
        }

        return (nullptr);
    }
};
// class ProblemFactory

}// namespace Plato

#endif /* PLATOPROBLEMFACTORY_HPP_ */
