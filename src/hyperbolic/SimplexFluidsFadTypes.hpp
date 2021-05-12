/*
 * SimplexFluidsFadTypes.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include <Sacado.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam SimplexPhysics physics type associated with simplex elements
 *
 * \struct SimplexFadTypes
 *
 * \brief The C++ structure owns the Forward Automatic Differentiation (FAD)
 * types used for the Quantities of Interest (QoI) in fluid flow applications.
 ******************************************************************************/
template<typename SimplexPhysics>
struct SimplexFadTypes
{
    using ConfigFad   = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumConfigDofsPerCell>;   /*!< configuration FAD type */
    using ControlFad  = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumNodesPerCell>;        /*!< control FAD type */
    using MassFad     = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMassDofsPerCell>;     /*!< mass FAD type */
    using EnergyFad   = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumEnergyDofsPerCell>;   /*!< energy FAD type */
    using MomentumFad = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMomentumDofsPerCell>; /*!< momentum FAD type */
};
// struct SimplexFadTypes

/***************************************************************************//**
 * \tparam SimplexFadTypesT physics type associated with simplex elements
 * \tparam ScalarType       scalar type
 *
 * \struct is_fad<SimplexFadTypesT, ScalarType>::value
 *
 * \brief is true if ScalarType is of any AD type defined in SimplexFadTypesT.
 ******************************************************************************/
template <typename SimplexFadTypesT, typename ScalarType>
struct is_fad {
  static constexpr bool value = std::is_same< ScalarType, typename SimplexFadTypesT::MassFad     >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::ControlFad  >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::ConfigFad   >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::EnergyFad   >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::MomentumFad >::value;
};
// struct is_fad


// which_fad<TypesT,T1,T2>::type returns:
// -- compile error  if T1 and T2 are both AD types defined in TypesT,
// -- T1             if only T1 is an AD type in TypesT,
// -- T2             if only T2 is an AD type in TypesT,
// -- T2             if neither are AD types.
//
template <typename TypesT, typename T1, typename T2>
struct which_fad {
  static_assert( !(is_fad<TypesT,T1>::value && is_fad<TypesT,T2>::value), "Only one template argument can be an AD type.");
  using type = typename std::conditional< is_fad<TypesT,T1>::value, T1, T2 >::type;
};


// fad_type_t<PhysicsT,T1,T2,T3,...,TN> returns:
// -- compile error  if more than one of T1,...,TN is an AD type in SimplexFadTypes<PhysicsT>,
// -- type TI        if only TI is AD type in SimplexFadTypes<PhysicsT>,
// -- TN             if none of TI are AD type in SimplexFadTypes<PhysicsT>.
//
template <typename TypesT, typename ...P> struct fad_type;
template <typename TypesT, typename T> struct fad_type<TypesT, T> { using type = T; };
template <typename TypesT, typename T, typename ...P> struct fad_type<TypesT, T, P ...> {
  using type = typename which_fad<TypesT, T, typename fad_type<TypesT, P...>::type>::type;
};
template <typename PhysicsT, typename ...P> using fad_type_t = typename fad_type<SimplexFadTypes<PhysicsT>,P...>::type;

/***************************************************************************//**
 *  \brief Base class for automatic differentiation types used in fluid problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam SimplexPhysicsT simplex fluid dynamic physics type
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr Plato::OrdinalType mNumSpatialDims        = SimplexPhysicsT::mNumSpatialDims;        /*!< number of spatial dimensions */
    static constexpr Plato::OrdinalType mNumNodesPerCell       = SimplexPhysicsT::mNumNodesPerCell;       /*!< number of nodes per simplex cell */
    static constexpr Plato::OrdinalType mNumControlDofsPerNode = SimplexPhysicsT::mNumControlDofsPerNode; /*!< number of design variable fields */
};
// struct EvaluationTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct ResultTypes
 *
 * \brief Scalar types for residual evaluations.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct ResultTypes : EvaluationTypes<SimplexPhysicsT>
{
    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = Plato::Scalar;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};
// struct ResultTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradCurrentMomentumTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the current momentum field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradCurrentMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = FadType;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};
// struct GradCurrentMomentumTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradCurrentEnergyTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the current energy field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradCurrentEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = FadType;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};
// struct GradCurrentEnergyTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradCurrentMassTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the current mass field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradCurrentMassTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MassFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = FadType;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};
// struct GradCurrentMassTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradPreviousMomentumTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the previous momentum field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradPreviousMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = FadType;

    using MomentumPredictorScalarType = Plato::Scalar;
};
// struct GradPreviousMomentumTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradPreviousEnergyTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the previous energy field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradPreviousEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = FadType;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};
// struct GradPreviousEnergyTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradPreviousMassTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the previous mass field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradPreviousMassTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MassFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = FadType;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};
// struct GradPreviousMassTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradMomentumPredictorTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the predictor field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradMomentumPredictorTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = FadType;
};
// struct GradMomentumPredictorTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradConfigTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to configuration variables.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradConfigTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = FadType;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};
// struct GradConfigTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradControlTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to control variables.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradControlTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::ControlFad;

    using ControlScalarType           = FadType;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = FadType;

    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;

    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;

    using MomentumPredictorScalarType = Plato::Scalar;
};
// struct GradControlTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct Evaluation
 *
 * \brief Wrapper structure for the evaluation types used in fluid flow applications.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct Evaluation
{
    using Residual         = ResultTypes<SimplexPhysicsT>;
    using GradConfig       = GradConfigTypes<SimplexPhysicsT>;
    using GradControl      = GradControlTypes<SimplexPhysicsT>;

    using GradCurMass      = GradCurrentMassTypes<SimplexPhysicsT>;
    using GradPrevMass     = GradPreviousMassTypes<SimplexPhysicsT>;

    using GradCurEnergy    = GradCurrentEnergyTypes<SimplexPhysicsT>;
    using GradPrevEnergy   = GradPreviousEnergyTypes<SimplexPhysicsT>;

    using GradCurMomentum  = GradCurrentMomentumTypes<SimplexPhysicsT>;
    using GradPrevMomentum = GradPreviousMomentumTypes<SimplexPhysicsT>;
    using GradPredictor    = GradMomentumPredictorTypes<SimplexPhysicsT>;
};
// struct Evaluation

}
// namespace Fluids

}
// namespace Plato
