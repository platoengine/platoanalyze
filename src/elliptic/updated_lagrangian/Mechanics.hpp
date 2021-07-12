#pragma once

#include <memory>

#include "elliptic/updated_lagrangian/SimplexMechanics.hpp"
#include "elliptic/updated_lagrangian/AbstractScalarFunction.hpp"
#include "elliptic/updated_lagrangian/InternalElasticEnergy.hpp"
#include "elliptic/updated_lagrangian/StressPNorm.hpp"
#include "elliptic/updated_lagrangian/ElastostaticResidual.hpp"

#include "AnalyzeMacros.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

namespace MechanicsFactory
{

/******************************************************************************//**
 * \brief Create elastostatics residual equation
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<EvaluationType>>
elastostatics_residual(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string              aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aProblemParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "NoPenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    return (tOutput);
}
// function elastostatics_residual


/******************************************************************************//**
 * \brief Create internal elastic energy criterion
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
 * \param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<EvaluationType>>
internal_elastic_energy(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aProblemParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::InternalElasticEnergy<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::InternalElasticEnergy<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::InternalElasticEnergy<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "NoPenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::InternalElasticEnergy<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function internal_elastic_energy


/******************************************************************************//**
 * \brief Create stress p-norm criterion
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze physics-based database
 * \param [in] aProblemParams input parameters
 * \param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<EvaluationType>>
stress_p_norm(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string            & aFuncName
)
{
    std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aProblemParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::StressPNorm<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::StressPNorm<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::StressPNorm<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tPenaltyType == "NoPenalty")
    {
        tOutput = std::make_shared<Plato::Elliptic::UpdatedLagrangian::StressPNorm<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    return (tOutput);
}
// function stress_p_norm


/******************************************************************************//**
 * @brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO vector function (i.e. residual equation)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze physics-based database
     * \param [in] aProblemParams input parameters
     * \param [in] aFuncName vector function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
              std::string              aFuncName)
    {

        if(aFuncName == "Updated Lagrangian Elliptic")
        {
            return (Plato::Elliptic::UpdatedLagrangian::MechanicsFactory::elastostatics_residual<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFuncName));
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList")
        }
    }


    /******************************************************************************//**
     * \brief Create a PLATO scalar function (i.e. optimization criterion)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze physics-based database
     * \param [in] aProblemParams input parameters
     * \param [in] aFuncType scalar function type
     * \param [in] aFuncName scalar function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
              std::string              aFuncType,
              std::string              aFuncName
    )
    {
        if(aFuncType == "Internal Elastic Energy")
        {
            return Plato::Elliptic::UpdatedLagrangian::MechanicsFactory::internal_elastic_energy<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(aFuncType == "Stress P-Norm")
        {
            return Plato::Elliptic::UpdatedLagrangian::MechanicsFactory::stress_p_norm<EvaluationType>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        {
            return nullptr;
        }
    }
};
// struct FunctionFactory

} // namespace MechanicsFactory

/******************************************************************************//**
 * \brief Concrete class for use as the SimplexPhysics template argument in
 *        Plato Problem
**********************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class Mechanics: public Plato::Elliptic::UpdatedLagrangian::SimplexMechanics<SpaceDimParam>
{
public:
    typedef Plato::Elliptic::UpdatedLagrangian::MechanicsFactory::FunctionFactory FunctionFactory;
    using SimplexT = SimplexMechanics<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato
