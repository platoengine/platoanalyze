#pragma once

#include "Simplex.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

#include "geometric/Volume.hpp"
#include "geometric/GeometryMisfit.hpp"
#include "geometric/AbstractScalarFunction.hpp"

namespace Plato {

namespace GeometryFactory {
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
struct FunctionFactory{
/******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Geometric::AbstractScalarFunction<EvaluationType>>
    createScalarFunction( 
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              aStrScalarFunctionType,
              std::string              aStrScalarFunctionName
    )
    {
        auto tLowerScalarFunc = Plato::tolower(aStrScalarFunctionType);
        if( tLowerScalarFunc == "volume" )
        {
            auto penaltyParams = aParamList.sublist("Criteria").sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = penaltyParams.get<std::string>("Type");
            auto tLowerPenaltyType = Plato::tolower(tPenaltyType);
            if( tLowerPenaltyType == "simp" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::MSIMP>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, aStrScalarFunctionName);
            }
            else
            if( tLowerPenaltyType == "ramp" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::RAMP>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, aStrScalarFunctionName);
            }
            else
            if( tLowerPenaltyType == "heaviside" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::Heaviside>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, aStrScalarFunctionName);
            }
            else
            {
                THROWERR(std::string("Unknown 'Penalty Function' of type '") + tLowerPenaltyType + "' specified in ParameterList");
            }
        } else
        if( tLowerScalarFunc == "geometry misfit" )
        {
            return std::make_shared<Plato::Geometric::GeometryMisfit<EvaluationType>>
               (aSpatialDomain, aDataMap, aParamList, aStrScalarFunctionName);
        }
        else
        {
            THROWERR(std::string("Unknown 'Objective' of type '") + tLowerScalarFunc + "' specified in 'Plato Problem' ParameterList");
        }
    }
};

} // namespace GeometryFactory

template <Plato::OrdinalType SpaceDimParam>
class Geometrical : public Plato::Simplex<SpaceDimParam> {
  public:
    typedef Plato::GeometryFactory::FunctionFactory<SpaceDimParam> FunctionFactory;
    using SimplexT = Simplex<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
    static constexpr Plato::OrdinalType mNumControl = 1;
};
// class Geometrical

} //namespace Plato
