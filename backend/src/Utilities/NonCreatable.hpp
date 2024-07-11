//==============================================================================
/*!
 *  From: file pe/util/NonCreatable.h
 *  \brief Base class for for non-creatable (static) classes
 *
 *  Copyright (C) 2009 Klaus Iglberger
 *
 *  This file is part of pe.
 *
 *  pe is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 *  pe is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with
 * pe. If not, see <http://www.gnu.org/licenses/>.
 */
//==============================================================================

#pragma once

namespace elastica {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Base class for non-creatable (static) classes.
  // \ingroup util
  //
  // The NonCreatable class is intended to work as a base class for
  // non-creatable classes, i.e. classes that cannot be instantiated and
  // exclusively offer static functions/data. Both the standard as well as the
  // copy constructor and the copy assignment operator are deleted
  // and left undefined in order to prohibit the instantiation of objects of
  // derived classes.
  //
  // \note
  // It is not necessary to publicly derive from this class. It is sufficient to
  // derive privately to prevent the instantiation of the derived class.
  //
  // \example
     \code
       class A : private NonCreatable
       { ... };
     \endcode
  //
  // \see NonCopyable
  */
  class NonCreatable {
   public:
    //**Constructors and copy assignment operator*******************************
    /*!\name Constructors and copy assignment operator */
    //@{
    //! Constructor (private & deleted)
    NonCreatable() = delete;
    //! Copy constructor (private & deleted)
    NonCreatable(const NonCreatable&) = delete;
    //! Move constructor (private & deleted)
    NonCreatable(NonCreatable&&) = delete;
    //! Copy assignment operator (private & deleted)
    NonCreatable& operator=(const NonCreatable&) = delete;
    //! Move assignment operator (private & deleted)
    NonCreatable& operator=(NonCreatable&&) = delete;
    //@}
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace elastica
