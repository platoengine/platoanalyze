#ifndef PLATO_HELMHOLTZ_HPP
#define PLATO_HELMHOLTZ_HPP

#include "Simplex.hpp"
#include "helmholtz/SimplexHelmholtz.hpp"

#include "helmholtz/AbstractVectorFunction.hpp"
#include "helmholtz/HelmholtzResidual.hpp"

namespace Plato {

namespace HelmholtzFactory {
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
struct FunctionFactory{
/******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Helmholtz::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              strVectorFunctionType
    )
    {

        if( strVectorFunctionType == "Helmholtz Filter" )
        {
            return std::make_shared<Plato::Helmholtz::HelmholtzResidual<EvaluationType>>
              (aSpatialDomain, aDataMap, aParamList);
        }
        else
        {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList for HELMHOLTZ FILTER");
        }
    }

};

} // namespace HelmholtzFactory

template <Plato::OrdinalType SpaceDimParam>
class HelmholtzFilter : public Plato::SimplexHelmholtz<SpaceDimParam> {
  public:
    typedef Plato::HelmholtzFactory::FunctionFactory<SpaceDimParam> FunctionFactory;
    using SimplexT = SimplexHelmholtz<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};
// class HelmholtzFilter

} //namespace Plato

#endif
