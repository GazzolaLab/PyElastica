__doc__ = """ Test modules for callback """
import numpy as np
import pytest

from elastica.callback_functions import CallBackBaseClass


class TestCallBacksClosing:
    from elastica.modules import BaseSystemCollection
    from elastica.modules import CallBacks

    class SystemCollectionWithCallBacksMixedin(BaseSystemCollection, CallBacks):
        pass

    def test_callback_closing_test_default_callback_impl(self):
        """
        Test if any class derived from CallBackBaseClass can be used
        without any error when simulator.close() is called.
        This is to check the backward compatibility, as many previous
        callback classes are derived from CallBackBaseClass,
        but does not have explicit implementation of on_close method.
        """
        sys_coll = self.SystemCollectionWithCallBacksMixedin()
        sys_coll.extend_allowed_types((int,))
        rod = 0

        class MockCallback(CallBackBaseClass):
            pass

        # build flag check for some MockCallback.on_close() function call

        sys_coll.append(rod)
        sys_coll.collect_diagnostics(rod).using(MockCallback)
        sys_coll.close()

    def test_callback_closing_custom(self):
        """
        Check if on_close is called properly with a custom callback.
        """
        sys_coll = self.SystemCollectionWithCallBacksMixedin()
        sys_coll.extend_allowed_types((int,))
        rod = 0

        CLOSE_CALLED_FLAG = []

        class MockCallback(CallBackBaseClass):
            def __init__(self, o):
                self.o = o

            def on_close(self):
                self.o.append(42)

        sys_coll.append(rod)
        sys_coll.collect_diagnostics(rod).using(MockCallback, o=CLOSE_CALLED_FLAG)
        sys_coll.close()

        # Before finalize, on_close function should not be hooked.
        assert not CLOSE_CALLED_FLAG

        # After finalize, on_close function should be called.
        sys_coll.finalize()
        sys_coll.close()

        assert len(CLOSE_CALLED_FLAG) == 1
        assert CLOSE_CALLED_FLAG[0] == 42
