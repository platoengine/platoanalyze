/*
 * FluidsUtils.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "BLAS1.hpp"
#include "UtilsOmegaH.hpp"
#include "UtilsTeuchos.hpp"
#include "PlatoUtilities.hpp"

namespace Plato
{

namespace Fluids
{

template<typename Type>
inline Type get_material_property(
    const std::string& aMaterialProperty,
    const std::string& aMaterialBlockName,
    Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Material Models") == false)
    {
        THROWERR("Parsing 'Plato Problem'. 'Material Models' parameter sublist not defined within 'Plato Problem' parameter list.")
    }
    auto tMaterialParamList = aInputs.sublist("Material Models");
    
    if(tMaterialParamList.isSublist(aMaterialBlockName) == false)
    {
        THROWERR(std::string("Parameter sublist with tag '") + aMaterialBlockName + "' is not defined within 'Material Models' Parameter List.")
    }
    auto tMyMaterial = tMaterialParamList.sublist(aMaterialBlockName);
    
    if(tMyMaterial.isParameter(aMaterialProperty) == false)
    {
        THROWERR(std::string("Requested material property with tag '") + aMaterialProperty 
            + "' is not defined in material Parameter Sublist '" + aMaterialBlockName 
            + " within the 'Material Models' Parameter List.")
    }
    auto tProperty = tMyMaterial.get<Type>(aMaterialProperty);
    return tProperty;
}

inline bool is_material_property_defined(
    const std::string& aMaterialProperty,
    const std::string& aMaterialBlockName,
    Teuchos::ParameterList& aInputs)
{
    if(aInputs.isSublist("Material Models") == false)
    {
        THROWERR("'Material Models' parameter list is not defined in the input file.")
    }
    auto tMaterialParamList = aInputs.sublist("Material Models");
    
    if(tMaterialParamList.isSublist(aMaterialBlockName) == false)
    {
        THROWERR(std::string("Material with tag '") + aMaterialBlockName + "' is not defined in 'Material Models' Parameter List.")
    }
    auto tMyMaterial = tMaterialParamList.sublist(aMaterialBlockName);
    
    return ( tMyMaterial.isParameter(aMaterialProperty) );
}

/***************************************************************************//**
 * \fn inline bool calculate_brinkman_forces
 *
 * \brief Return true if Brinkman forces are enabled, return false if disabled.
 * \param [in] aInputs input file metadata
 * \return boolean (true or false)
 ******************************************************************************/
inline bool calculate_brinkman_forces
(Teuchos::ParameterList& aInputs)
{
    if (aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tFlowProps = aInputs.sublist("Hyperbolic");
    auto tScenario = tFlowProps.get<std::string>("Scenario", "Analysis");
    auto tLowerScenario = Plato::tolower(tScenario);
    if (tLowerScenario == "density-based topology optimization")
    {
        return true;
    }
    return false;
}
// function calculate_brinkman_forces

/***************************************************************************//**
 * \fn inline std::string heat_transfer_tag
 *
 * \brief Parse heat transfer mechanism tag from input file and convert value to lowercase.
 * \param [in] aInputs input file metadata
 * \return lowercase heat transfer mechanism tag
 ******************************************************************************/
inline std::string heat_transfer_tag
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
    auto tHeatTransfer = Plato::tolower(tTag);
    return tHeatTransfer;
}
// function heat_transfer_tag

/***************************************************************************//**
 * \fn inline bool calculate_heat_transfer
 *
 * \brief Returns true if energy equation is enabled, else, returns false.
 * \param [in] aInputs input file metadata
 * \return boolean (true or false)
 ******************************************************************************/
inline bool calculate_heat_transfer
(Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;
    return tCalculateHeatTransfer;
}
// function calculate_heat_transfer

/***************************************************************************//**
 * \fn inline bool calculate_effective_conductivity
 *
 * \brief Calculate effective conductivity based on the heat transfer mechanism requested.
 * \param [in] aMaterialName material name for a given spatial domain (i.e. element block)
 * \param [in] aInputs input file metadata
 * \return effective conductivity
 ******************************************************************************/
inline Plato::Scalar
calculate_effective_conductivity
(const std::string& aMaterialName,
 Teuchos::ParameterList & aInputs)
{
    auto tOutput = 0;
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if(tHeatTransfer == "forced" || tHeatTransfer == "mixed")
    {
        auto tPrNum = Plato::Fluids::get_material_property<Plato::Scalar>("Prandtl Number", aMaterialName, aInputs);
        auto tReNum = Plato::Fluids::get_material_property<Plato::Scalar>("Reynolds Number", aMaterialName, aInputs);
        tOutput = static_cast<Plato::Scalar>(1) / (tReNum*tPrNum);
    }
    else if(tHeatTransfer == "natural")
    {
        tOutput = 1.0;
    }
    else
    {
        THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
    }
    return tOutput;
}
// function calculate_effective_conductivity

/***************************************************************************//**
 * \fn inline Plato::Scalar calculate_viscosity_constant
 *
 * \brief Calculate dimensionless viscocity \f$ \nu f\$ constant. The dimensionless
 * viscocity is given by \f$ \nu=\frac{1}{Re} f\$ if forced convection dominates or
 * by \f$ \nu=Pr \f$ is natural convection dominates.
 *
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless viscocity
 ******************************************************************************/
inline Plato::Scalar
calculate_viscosity_constant
(const std::string& aMaterialName,
 Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if(tHeatTransfer == "forced" || tHeatTransfer == "mixed" || tHeatTransfer == "none")
    {
        auto tReNum = Plato::Fluids::get_material_property<Plato::Scalar>("Reynolds Number", aMaterialName, aInputs);
        auto tViscocity = static_cast<Plato::Scalar>(1) / tReNum;
        return tViscocity;
    }
    else if(tHeatTransfer == "natural")
    {
        auto tViscocity = Plato::Fluids::get_material_property<Plato::Scalar>("Prandtl Number", aMaterialName, aInputs);
        return tViscocity;
    }
    else
    {
        THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
    }
}
// function calculate_viscosity_constant

/***************************************************************************//**
 * \fn inline Plato::Scalar buoyancy_constant_mixed_convection_problems
 *
 * \brief Calculate buoyancy constant for mixed convection problems.
 *
 * \param [in] aInputs input file metadata
 * \param [in] aMaterialName material model name for this spatial domain
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
buoyancy_constant_mixed_convection_problems
(const std::string& aMaterialName,
 Teuchos::ParameterList& aInputs)
{
    if( Plato::Fluids::is_material_property_defined("Richardson Number", aMaterialName, aInputs) )
    {
        return static_cast<Plato::Scalar>(1.0);
    }
    else if( Plato::Fluids::is_material_property_defined("Grashof Number", aMaterialName, aInputs) )
    {
        auto tReNum = Plato::Fluids::get_material_property<Plato::Scalar>("Reynolds Number", aMaterialName, aInputs); 
        auto tBuoyancy = static_cast<Plato::Scalar>(1.0) / (tReNum * tReNum);
        return tBuoyancy;
    }
    else
    {
        THROWERR("Mixed convection properties are not defined. One of these two options should be provided: 'Grashof Number' or 'Richardson Number'")
    }
}
// function buoyancy_constant_mixed_convection_problems

/***************************************************************************//**
 * \fn inline Plato::Scalar buoyancy_constant_natural_convection_problems
 *
 * \brief Calculate buoyancy constant for natural convection problems.
 *
 * \param [in] aInputs input file metadata
 * \param [in] aMaterialName material name for a given spatial domain
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
buoyancy_constant_natural_convection_problems
(const std::string& aMaterialName,
 Teuchos::ParameterList & aInputs)
{
    auto tPrNum = Plato::Fluids::get_material_property<Plato::Scalar>("Prandtl Number", aMaterialName, aInputs); 
    if( Plato::Fluids::is_material_property_defined("Rayleigh Number", aMaterialName, aInputs) )
    {
        auto tBuoyancy = tPrNum;
        return tBuoyancy;
    }
    else if( Plato::Fluids::is_material_property_defined("Grashof Number", aMaterialName, aInputs) )
    {
        auto tBuoyancy = tPrNum*tPrNum;
        return tBuoyancy;
    }
    else
    {
        THROWERR("Natural convection properties are not defined. One of these two options should be provided: 'Grashof Number' or 'Rayleigh Number'")
    }
}
// function buoyancy_constant_natural_convection_problems

/***************************************************************************//**
 * \fn inline Plato::Scalar calculate_buoyancy_constant
 *
 * \brief Calculate dimensionless buoyancy constant \f$ \beta f\$. The buoyancy
 * constant is defined by \f$ \beta=\frac{1}{Re^2} f\$ if forced convection dominates.
 * In contrast, the buoyancy constant for natural convection dominated problems
 * is given by \f$ \nu=Pr^2 \f$ or \f$ \nu=Pr \f$ depending on which dimensionless
 * convective constant was provided by the user (e.g. Rayleigh or Grashof number).
 *
 * \param [in] aMaterialName material model name for a given spatial domain
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
calculate_buoyancy_constant
(const std::string& aMaterialName, 
 Teuchos::ParameterList & aInputs)
{
    Plato::Scalar tBuoyancy = 0.0; // heat transfer calculations inactive if buoyancy = 0.0

    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if(tHeatTransfer == "mixed")
    {
        tBuoyancy = Plato::Fluids::buoyancy_constant_mixed_convection_problems(aMaterialName, aInputs);
    }
    else if(tHeatTransfer == "natural")
    {
        tBuoyancy = Plato::Fluids::buoyancy_constant_natural_convection_problems(aMaterialName, aInputs);
    }
    else if(tHeatTransfer == "forced" || tHeatTransfer == "none")
    {
        tBuoyancy = 0.0;
    }
    else
    {
        THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
    }

    return tBuoyancy;
}
// function calculate_buoyancy_constant

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector rayleigh_number
 *
 * \brief Parse dimensionless Rayleigh constants.
 *
 * \param [in] aMaterialName material name for a given spatial domain (i.e. element block).
 * \param [in] aInputs input file metadata
 *
 * \return Rayleigh constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
rayleigh_number
(const std::string& aMaterialName, 
 Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

    Plato::ScalarVector tOuput("Rayleigh Number", SpaceDim);
    if(tCalculateHeatTransfer)
    {
        auto tRaNum = Plato::Fluids::get_material_property< Teuchos::Array<Plato::Scalar> >("Rayleigh Number", aMaterialName, aInputs);
        if(tRaNum.size() != SpaceDim)
        {
            THROWERR(std::string("'Rayleigh Number' array length should match the number of spatial dimensions. ")
                + "Array length is '" + std::to_string(tRaNum.size()) + "' and the number of spatial dimensions is '"
                + std::to_string(SpaceDim) + "'.")
        }

        auto tHostRaNum = Kokkos::create_mirror(tOuput);
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            tHostRaNum(tDim) = tRaNum[tDim];
        }
        Kokkos::deep_copy(tOuput, tHostRaNum);
    }
    else
    {
        Plato::blas1::fill(0.0, tOuput);
    }

    return tOuput;
}
// function rayleigh_number

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector grashof_number
 *
 * \brief Parse dimensionless Grashof constants.
 * 
 * \param [in] aMaterialName material name for a given spatial domain (e.g. element block)
 * \param [in] aInputs input file metadata
 *
 * \return Grashof constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
grashof_number
(const std::string& aMaterialName,
 Teuchos::ParameterList& aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

    Plato::ScalarVector tOuput("Grashof Number", SpaceDim);
    if(tCalculateHeatTransfer)
    {
        auto tGrNum = Plato::Fluids::get_material_property< Teuchos::Array<Plato::Scalar> >("Grashof Number", aMaterialName, aInputs);
        if(tGrNum.size() != SpaceDim)
        {
            THROWERR(std::string("'Grashof Number' array length should match the number of spatial dimensions. ")
                + "Array length is '" + std::to_string(tGrNum.size()) + "' and the number of spatial dimensions is '"
                + std::to_string(SpaceDim) + "'.")
        }

        auto tHostGrNum = Kokkos::create_mirror(tOuput);
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            tHostGrNum(tDim) = tGrNum[tDim];
        }
        Kokkos::deep_copy(tOuput, tHostGrNum);
    }
    else
    {
        Plato::blas1::fill(0.0, tOuput);
    }

    return tOuput;
}
// function grashof_number

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector richardson_number
 *
 * \brief Parse dimensionless Richardson constants.
 *
 * \param [in] aMaterialName material name for a given spatial domain (e.g. element block)
 * \param [in] aInputs input file metadata
 *
 * \return Richardson constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
richardson_number
(const std::string& aMaterialName,
 Teuchos::ParameterList& aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

    Plato::ScalarVector tOuput("Grashof Number", SpaceDim);
    if(tCalculateHeatTransfer)
    {
        auto tRiNum = Plato::Fluids::get_material_property< Teuchos::Array<Plato::Scalar> >("Richardson Number", aMaterialName, aInputs);
        if(tRiNum.size() != SpaceDim)
        {
            THROWERR(std::string("'Richardson Number' array length should match the number of spatial dimensions. ")
                + "Array length is '" + std::to_string(tRiNum.size()) + "' and the number of spatial dimensions is '"
                + std::to_string(SpaceDim) + "'.")
        }

        auto tHostRiNum = Kokkos::create_mirror(tOuput);
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            tHostRiNum(tDim) = tRiNum[tDim];
        }
        Kokkos::deep_copy(tOuput, tHostRiNum);
    }
    else
    {
        Plato::blas1::fill(0.0, tOuput);
    }

    return tOuput;
}
// function richardson_number

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector parse_natural_convection_number
 *
 * \brief Parse dimensionless natural convection constants (e.g. Rayleigh or Grashof number).
 *
 * \param [in] aInputs input file metadata
 *
 * \return natural convection constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
parse_natural_convection_number
(const std::string& aMaterialName, 
 Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if( Plato::Fluids::is_material_property_defined("Rayleigh Number", aMaterialName, aInputs) &&
            (tHeatTransfer == "natural") )
    {
        return (Plato::Fluids::rayleigh_number<SpaceDim>(aMaterialName, aInputs));
    }
    else if( Plato::Fluids::is_material_property_defined("Grashof Number", aMaterialName, aInputs) &&
            (tHeatTransfer == "natural" || tHeatTransfer == "mixed") )
    {
        return (Plato::Fluids::grashof_number<SpaceDim>(aMaterialName, aInputs));
    }
    else if( Plato::Fluids::is_material_property_defined("Richardson Number", aMaterialName, aInputs) &&
            (tHeatTransfer == "mixed") )
    {
        return (Plato::Fluids::richardson_number<SpaceDim>(aMaterialName, aInputs));
    }
    else
    {
        THROWERR(std::string("Natural convection properties are not defined. One of these options") +
                 " should be provided: 'Grashof Number' (for natural or mixed convection problems), " +
                 "'Rayleigh Number' (for natural convection problems), or 'Richardson Number' (for mixed convection problems).")
    }
}
// function parse_natural_convection_number

/***************************************************************************//**
 * \fn inline Plato::Scalar stabilization_constant
 *
 * \brief Parse stabilization force scalar multiplier.
 *
 * \param [in] aSublistName parameter sublist name
 * \param [in] aInputs      input file metadata
 *
 * \return scalar multiplier
 ******************************************************************************/
inline Plato::Scalar
stabilization_constant
(const std::string & aSublistName,
 Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tOutput = 0.0;
    auto tFlowProps = aInputs.sublist("Hyperbolic");
    if(tFlowProps.isSublist(aSublistName))
    {
        auto tMomentumConservation = tFlowProps.sublist(aSublistName);
        tOutput = tMomentumConservation.get<Plato::Scalar>("Stabilization Constant", 0.0);
    }
    return tOutput;
}
// function stabilization_constant

/******************************************************************************//**
 * \fn inline Plato::ScalarVector calculate_characteristic_element_size
 *
 * \tparam NumSpatialDims  spatial dimensions (integer)
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 *
 * \brief Calculate characteristic size for all the elements on the finite element mesh.
 *
 * \param [in] aModel spatial model database, holds such as mesh information.
 * \return array of element characteristic size
 *
 **********************************************************************************/
template
<Plato::OrdinalType NumSpatialDims,
 Plato::OrdinalType NumNodesPerCell>
inline Plato::ScalarVector
calculate_characteristic_element_size
(const Plato::SpatialModel & aModel)
{
    auto tCoords = aModel.Mesh.coords();
    auto tCells2Nodes = aModel.Mesh.ask_elem_verts();

    Plato::OrdinalType tNumCells = aModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aModel.Mesh.nverts();
    Plato::ScalarVector tElemCharSize("element characteristic size", tNumNodes);
    Plato::blas1::fill(std::numeric_limits<Plato::Scalar>::max(), tElemCharSize);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tElemSize = Plato::omega_h::calculate_element_size<NumSpatialDims,NumNodesPerCell>(aCellOrdinal, tCells2Nodes, tCoords);
        for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
        {
            auto tVertexIndex = tCells2Nodes[aCellOrdinal*NumNodesPerCell + tNode];
            tElemCharSize(tVertexIndex) = tElemSize <= tElemCharSize(tVertexIndex) ? tElemSize : tElemCharSize(tVertexIndex);
        }
    },"calculate characteristic element size");

    return tElemCharSize;
}
// function calculate_characteristic_element_size

/******************************************************************************//**
 * \fn inline Plato::ScalarVector calculate_magnitude_convective_velocity
 *
 * \tparam NodesPerCell number of nodes per cell (integer)
 *
 * \brief Calculate convective velocity magnitude at each node.
 *
 * \param [in] aModel    spatial model database, holds such as mesh information
 * \param [in] aVelocity velocity field
 *
 * \return convective velocity magnitude at each node
 *
 **********************************************************************************/
template<Plato::OrdinalType NodesPerCell>
Plato::ScalarVector
calculate_magnitude_convective_velocity
(const Plato::SpatialModel & aModel,
 const Plato::ScalarVector & aVelocity)
{
    auto tCell2Node = aModel.Mesh.ask_elem_verts();
    Plato::OrdinalType tSpaceDim = aModel.Mesh.dim();
    Plato::OrdinalType tNumCells = aModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aModel.Mesh.nverts();

    Plato::ScalarVector tConvectiveVelocity("convective velocity", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCell)
    {
        for(Plato::OrdinalType tNode = 0; tNode < NodesPerCell; tNode++)
        {
            Plato::Scalar tSum = 0.0;
            Plato::OrdinalType tVertexIndex = tCell2Node[aCell*NodesPerCell + tNode];
            for(Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
            {
                auto tDofIndex = tVertexIndex * tSpaceDim + tDim;
                tSum += aVelocity(tDofIndex) * aVelocity(tDofIndex);
            }
            auto tMyValue = sqrt(tSum);
            tConvectiveVelocity(tVertexIndex) =
                tMyValue >= tConvectiveVelocity(tVertexIndex) ? tMyValue : tConvectiveVelocity(tVertexIndex);
        }
    }, "calculate_magnitude_convective_velocity");

    return tConvectiveVelocity;
}
// function calculate_magnitude_convective_velocity

/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_critical_diffusion_time_step
 *
 * \brief Calculate critical diffusion time step.
 *
 * \param [in] aKinematicViscocity kinematic viscocity
 * \param [in] aThermalDiffusivity thermal diffusivity
 * \param [in] aCharElemSize       characteristic element size
 * \param [in] aSafetyFactor       safety factor
 *
 * \return critical diffusive time step scalar
 *
 **********************************************************************************/
inline Plato::Scalar
calculate_critical_diffusion_time_step
(const Plato::Scalar aKinematicViscocity,
 const Plato::Scalar aThermalDiffusivity,
 const Plato::ScalarVector & aCharElemSize,
 Plato::Scalar aSafetyFactor = 0.7)
{
    auto tNumNodes = aCharElemSize.size();
    Plato::ScalarVector tLocalTimeStep("time step", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        auto tKinematicStep = ( aSafetyFactor * aCharElemSize(aNodeOrdinal) * aCharElemSize(aNodeOrdinal) ) /
                ( static_cast<Plato::Scalar>(2) * aKinematicViscocity );
        auto tDiffusivityStep = ( aSafetyFactor * aCharElemSize(aNodeOrdinal) * aCharElemSize(aNodeOrdinal) ) /
                ( static_cast<Plato::Scalar>(2) * aThermalDiffusivity );
        tLocalTimeStep(aNodeOrdinal) = tKinematicStep < tDiffusivityStep ? tKinematicStep : tDiffusivityStep;
    }, "calculate local critical time step");

    Plato::Scalar tMinValue = 0.0;
    Plato::blas1::min(tLocalTimeStep, tMinValue);
    return tMinValue;
}
// function calculate_critical_diffusion_time_step

/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_critical_time_step_upper_bound
 *
 * \brief Calculate critical time step upper bound.
 *
 * \param [in] aVelUpperBound critical velocity lower bound
 * \param [in] aCharElemSize  characteristic element size
 *
 * \return critical time step upper bound (scalar)
 *
 **********************************************************************************/
inline Plato::Scalar
calculate_critical_time_step_upper_bound
(const Plato::Scalar aVelUpperBound,
 const Plato::ScalarVector& aCharElemSize)
{
    Plato::Scalar tMinValue = 0.0;
    Plato::blas1::min(aCharElemSize, tMinValue);
    auto tOutput = tMinValue / aVelUpperBound;
    return tOutput;
}
// function calculate_critical_time_step_upper_bound

/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_critical_convective_time_step
 *
 * \brief Calculate critical convective time step.
 *
 * \param [in] aModel spatial model metadata
 * \param [in] aCharElemSize  characteristic element size
 * \param [in] aVelocity      velocity field
 * \param [in] aSafetyFactor  safety factor multiplier (default = 0.7)
 *
 * \return critical convective time step (scalar)
 *
 **********************************************************************************/
inline Plato::Scalar
calculate_critical_convective_time_step
(const Plato::SpatialModel & aModel,
 const Plato::ScalarVector & aCharElemSize,
 const Plato::ScalarVector & aVelocity,
 Plato::Scalar aSafetyFactor = 0.7)
{
    auto tNorm = Plato::blas1::norm(aVelocity);
    if(tNorm <= std::numeric_limits<Plato::Scalar>::min())
    {
        return std::numeric_limits<Plato::Scalar>::max();
    }

    auto tNumNodes = aModel.Mesh.nverts();
    Plato::ScalarVector tLocalTimeStep("time step", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tLocalTimeStep(aNodeOrdinal) = aSafetyFactor * ( aCharElemSize(aNodeOrdinal) / aVelocity(aNodeOrdinal) );
    }, "calculate local critical time step");

    Plato::Scalar tMinValue = 0;
    Plato::blas1::min(tLocalTimeStep, tMinValue);
    return tMinValue;
}
// function calculate_critical_convective_time_step

}
// namespace Fluids

}
//namespace Plato
