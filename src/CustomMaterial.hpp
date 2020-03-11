#ifndef CUSTOMMATERIAL_HPP
#define CUSTOMMATERIAL_HPP

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************/
/*!
  \brief Class for custom material models
*/
class CustomMaterial
/******************************************************************************/
{
public:
    CustomMaterial(const Teuchos::ParameterList& aParamList) {}
    virtual ~CustomMaterial() {}

protected:
    virtual double GetCustomExpressionValue( const Teuchos::ParameterList& paramList);
};
// class CustomMaterial

} // namespace Plato

#endif
