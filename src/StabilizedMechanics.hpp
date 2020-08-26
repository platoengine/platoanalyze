#ifndef PLATO_STABILIZED_MECHANICS_HPP
#define PLATO_STABILIZED_MECHANICS_HPP

#include <memory>

// TODO needed? #include <Omega_h_mesh.hpp>
// TODO needed? #include <Omega_h_assoc.hpp>

#include "parabolic/AbstractScalarFunction.hpp"

#include "elliptic/AbstractScalarFunction.hpp"

#include "SimplexProjection.hpp"
#include "SimplexStabilizedMechanics.hpp"
#include "StabilizedElastostaticResidual.hpp"
#include "StabilizedElastostaticEnergy.hpp"
#include "Plasticity.hpp"
#include "AnalyzeMacros.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace StabilizedMechanicsFactory
{

/******************************************************************************//**
 * \brief Create elastostatics residual equation
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap PLATO Analyze physics-based database
 * \param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractVectorFunctionVMS<EvaluationType>>
stabilized_elastostatics_residual(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aInputParams,
    const std::string            & aFuncName
)
{
    std::shared_ptr<AbstractVectorFunctionVMS<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::StabilizedElastostaticResidual<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::StabilizedElastostaticResidual<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::StabilizedElastostaticResidual<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "NoPenalty")
    {
        tOutput = std::make_shared<Plato::StabilizedElastostaticResidual<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aInputParams, tPenaltyParams);
    }
    return (tOutput);
}
// function stabilized_elastostatics_residual


/******************************************************************************//**
 * \brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO vector function (i.e. residual equation)
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Analyze physics-based database
     * \param [in] aInputParams input parameters
     * \param [in] aFuncName vector function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    )
    {
        if(aFuncName == "Elliptic")
        {
            return (Plato::StabilizedMechanicsFactory::stabilized_elastostatics_residual<EvaluationType>
                     (aSpatialDomain, aDataMap, aInputParams, aFuncName));
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
        const std::string            & aFuncType,
        const std::string            & aFuncName
    )
    /******************************************************************************/
    {
        std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> tOutput;
        auto tPenaltyParams = aParamList.sublist(aFuncName).sublist("Penalty Function");
        if( aFuncType == "Internal Elastic Energy" )
        {
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                tOutput = std::make_shared<Plato::StabilizedElastostaticEnergy<EvaluationType, Plato::MSIMP>>
                            (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aFuncName);
            }
            else
            if(tPenaltyType == "RAMP")
            {
                tOutput = std::make_shared<Plato::StabilizedElastostaticEnergy<EvaluationType, Plato::RAMP>>
                            (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aFuncName);
            }
            else
            if(tPenaltyType == "Heaviside")
            {
                tOutput = std::make_shared<Plato::StabilizedElastostaticEnergy<EvaluationType, Plato::Heaviside>>
                            (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aFuncName);
            }
            else
            if(tPenaltyType == "NoPenalty")
            {
                tOutput = std::make_shared<Plato::StabilizedElastostaticEnergy<EvaluationType, Plato::NoPenalty>>
                            (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aFuncName);
            }
        }
        else
        {
            THROWERR("Unknown scalar function specified in 'Plato Problem' ParameterList");
        }
        return (tOutput);
    }
};
// struct FunctionFactory

} // namespace StabilizedMechanicsFactory

/****************************************************************************//**
 * \brief Concrete class for use as the SimplexPhysics template argument in
 *        EllipticVMSProblem
 *******************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class StabilizedMechanics : public Plato::SimplexStabilizedMechanics<SpaceDimParam>
{
public:
    typedef Plato::StabilizedMechanicsFactory::FunctionFactory FunctionFactory;

    using SimplexT    = SimplexStabilizedMechanics<SpaceDimParam>;
    using ProjectorT  = typename Plato::Projection<SpaceDimParam, SimplexT::mNumDofsPerNode, SimplexT::mPressureDofOffset, /* numProjectionDofs=*/ 1>;

    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};
// class StabilizedMechanics

}
// namespace Plato

#endif
