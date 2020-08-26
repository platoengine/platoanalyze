#ifndef PLATO_PROJECTION_HPP
#define PLATO_PROJECTION_HPP

#include <memory>

// TODO neeeded? #include <Omega_h_mesh.hpp>
// TODO neeeded? #include <Omega_h_assoc.hpp>

#include "SimplexProjection.hpp"
#include "PressureGradientProjectionResidual.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace ProjectionFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aParamList, 
        const std::string            & aStrVectorFunctionType
    )
    /******************************************************************************/
    {
        if(aStrVectorFunctionType == "State Gradient Projection")
        {
            auto tPenaltyParams = aParamList.sublist(aStrVectorFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::PressureGradientProjectionResidual<EvaluationType, Plato::MSIMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::PressureGradientProjectionResidual<EvaluationType, Plato::RAMP>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::PressureGradientProjectionResidual<EvaluationType, Plato::Heaviside>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::PressureGradientProjectionResidual<EvaluationType, Plato::NoPenalty>>
                         (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
}; // struct FunctionFactory

} // namespace ProjectionFactory


/****************************************************************************//**
 * \brief Concrete class for use as the PhysicsT template argument in VectorFunctionVMS
 *******************************************************************************/
template<
  Plato::OrdinalType SpaceDimParam,
  Plato::OrdinalType TotalDofsParam = SpaceDimParam,
  Plato::OrdinalType ProjectionDofOffset = 0,
  Plato::OrdinalType NumProjectionDofs = 1>
class Projection: public Plato::SimplexProjection<SpaceDimParam>
{
public:
    typedef Plato::ProjectionFactory::FunctionFactory FunctionFactory;
    using SimplexT = SimplexProjection<SpaceDimParam, TotalDofsParam, ProjectionDofOffset, NumProjectionDofs>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

} // namespace Plato

#endif
