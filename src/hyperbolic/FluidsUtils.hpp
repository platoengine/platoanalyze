/*
 * FluidsUtils.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "BLAS1.hpp"
#include "UtilsTeuchos.hpp"
#include "PlatoUtilities.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \fn inline bool is_dimensionless_parameter_defined
 *
 * \brief Check if dimensionless parameter is deifned.
 *
 * \param [in] aTag    parameter tag
 * \param [in] aInputs input file metadata
 *
 * \return boolean (true or false)
 ******************************************************************************/
inline bool is_dimensionless_parameter_defined
(const std::string & aTag,
 Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    if( tHyperbolic.isSublist("Dimensionless Properties") == false )
    {
        THROWERR("'Dimensionless Properties' sublist is not defined.")
    }
    auto tSublist = tHyperbolic.sublist("Dimensionless Properties");
    auto tIsDefined = tSublist.isParameter(aTag);
    return tIsDefined;
}
// function is_dimensionless_parameter_defined

/***************************************************************************//**
 * \fn inline Plato::Scalar reynolds_number
 *
 * \brief Parse Reynolds number from input file.
 * \param [in] aInputs input file metadata
 * \return Reynolds number
 ******************************************************************************/
inline Plato::Scalar
reynolds_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tReNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
    return tReNum;
}
// function reynolds_number

/***************************************************************************//**
 * \fn inline Plato::Scalar prandtl_number
 *
 * \brief Parse Prandtl number from input file.
 * \param [in] aInputs input file metadata
 * \return Prandtl number
 ******************************************************************************/
inline Plato::Scalar
prandtl_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tPrNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
    return tPrNum;
}
// function prandtl_number

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
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tScenario = tHyperbolic.get<std::string>("Scenario", "Analysis");
    auto tLowerScenario = Plato::tolower(tScenario);
    if(tLowerScenario == "density to")
    {
    return true;
    }
    return false;
}
// function calculate_brinkman_forces

/***************************************************************************//**
 * \fn inline std::string heat_transfer_tag
 *
 * \brief Parse heat transfer mechanism tag from input file.
 * \param [in] aInputs input file metadata
 * \return heat transfer mechanism tag
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
 * \param [in] aInputs input file metadata
 * \return effective conductivity
 ******************************************************************************/
inline Plato::Scalar
calculate_effective_conductivity
(Teuchos::ParameterList & aInputs)
{
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "natural");
    auto tHeatTransfer = Plato::tolower(tTag);

    auto tOutput = 0;
    if(tHeatTransfer == "forced" || tHeatTransfer == "mixed")
    {
        auto tPrNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
        auto tReNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
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
(Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if(tHeatTransfer == "forced" || tHeatTransfer == "mixed" || tHeatTransfer == "none")
    {
        auto tReNum = Plato::Fluids::reynolds_number(aInputs);
        auto tViscocity = static_cast<Plato::Scalar>(1) / tReNum;
        return tViscocity;
    }
    else if(tHeatTransfer == "natural")
    {
        auto tViscocity = Plato::Fluids::prandtl_number(aInputs);
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
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
buoyancy_constant_mixed_convection_problems
(Teuchos::ParameterList & aInputs)
{
    if(Plato::Fluids::is_dimensionless_parameter_defined("Richardson Number", aInputs))
    {
        return static_cast<Plato::Scalar>(1.0);
    }
    else if(Plato::Fluids::is_dimensionless_parameter_defined("Grashof Number", aInputs))
    {
        auto tReNum = Plato::Fluids::reynolds_number(aInputs);
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
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
buoyancy_constant_natural_convection_problems
(Teuchos::ParameterList & aInputs)
{
    auto tPrNum = Plato::Fluids::prandtl_number(aInputs);
    if(Plato::Fluids::is_dimensionless_parameter_defined("Rayleigh Number", aInputs))
    {
        auto tBuoyancy = tPrNum;
        return tBuoyancy;
    }
    else if(Plato::Fluids::is_dimensionless_parameter_defined("Grashof Number", aInputs))
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
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
calculate_buoyancy_constant
(Teuchos::ParameterList & aInputs)
{
    Plato::Scalar tBuoyancy = 0.0; // heat transfer calculations inactive if buoyancy = 0.0

    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if(tHeatTransfer == "mixed")
    {
        tBuoyancy = Plato::Fluids::buoyancy_constant_mixed_convection_problems(aInputs);
    }
    else if(tHeatTransfer == "natural")
    {
        tBuoyancy = Plato::Fluids::buoyancy_constant_natural_convection_problems(aInputs);
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
 * \param [in] aInputs input file metadata
 *
 * \return Rayleigh constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
rayleigh_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
    auto tHeatTransfer = Plato::tolower(tTag);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

    Plato::ScalarVector tOuput("Rayleigh Number", SpaceDim);
    if(tCalculateHeatTransfer)
    {
        auto tRaNum = Plato::teuchos::parse_parameter<Teuchos::Array<Plato::Scalar>>("Rayleigh Number", "Dimensionless Properties", tHyperbolic);
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
 * \param [in] aInputs input file metadata
 *
 * \return Grashof constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
grashof_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
    auto tHeatTransfer = Plato::tolower(tTag);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

    Plato::ScalarVector tOuput("Grashof Number", SpaceDim);
    if(tCalculateHeatTransfer)
    {
        auto tGrNum = Plato::teuchos::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof Number", "Dimensionless Properties", tHyperbolic);
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
 * \param [in] aInputs input file metadata
 *
 * \return Richardson constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
richardson_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
    auto tHeatTransfer = Plato::tolower(tTag);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

    Plato::ScalarVector tOuput("Grashof Number", SpaceDim);
    if(tCalculateHeatTransfer)
    {
        auto tRiNum = Plato::teuchos::parse_parameter<Teuchos::Array<Plato::Scalar>>("Richardson Number", "Dimensionless Properties", tHyperbolic);
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
(Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if( Plato::Fluids::is_dimensionless_parameter_defined("Rayleigh Number", aInputs) &&
            (tHeatTransfer == "natural") )
    {
        return (Plato::Fluids::rayleigh_number<SpaceDim>(aInputs));
    }
    else if( Plato::Fluids::is_dimensionless_parameter_defined("Grashof Number", aInputs) &&
            (tHeatTransfer == "natural" || tHeatTransfer == "mixed") )
    {
        return (Plato::Fluids::grashof_number<SpaceDim>(aInputs));
    }
    else if( Plato::Fluids::is_dimensionless_parameter_defined("Richardson Number", aInputs) &&
            (tHeatTransfer == "mixed") )
    {
        return (Plato::Fluids::richardson_number<SpaceDim>(aInputs));
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
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    if(tHyperbolic.isSublist(aSublistName))
    {
        auto tMomentumConservation = tHyperbolic.sublist(aSublistName);
        tOutput = tMomentumConservation.get<Plato::Scalar>("Stabilization Constant", 0.0);
    }
    return tOutput;
}
// function stabilization_constant

}

}
