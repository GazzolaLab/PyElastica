## Configuration

Contains all default-settings to make a default `Elastica++` simulation. The type customizations are defined as macros (as
opposed to type-aliases) to avoid including header files and complicating the editing of these files. Any numeric
customizations
(such as setting a threshold) is done using `constexpr` functions.

These are later included as proper type-aliases inside ModuleSettings (with appropriate type and configuration checking)
.

### !!NOTE!!
Only edit these when you feel like the default configuration that you want for `Elastica++` differs from the defaults set in
the Github repository. Else prefer making a configuration in the main.cpp file, as this promotes readability and intent
to the poor souls (made to) read(ing) your code.
