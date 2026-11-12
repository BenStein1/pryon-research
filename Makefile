PYTHON ?= python3
CC ?= cc
CFLAGS := -O2 -Wall -Wextra -ffp-contract=off
SUPPORT := native/support
.PHONY: test demo math-check cross

test: math-check
	$(PYTHON) -m unittest discover -s tests -v

demo:
	$(PYTHON) demo.py

build/pryon-math-check: native/pryon_qat_test.c native/pryon_qat.c native/pryon_qat.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Wno-misleading-indentation $< native/pryon_qat.c -lm -o $@

math-check: build/pryon-math-check
	./build/pryon-math-check

build/pryon-train: native/pryon_train.c native/pryon_qat.c native/pryon_qat.h $(SUPPORT)/json.c $(SUPPORT)/hash.c
	@mkdir -p build
	$(CC) $(CFLAGS) -O3 -Wno-misleading-indentation -I$(SUPPORT) native/pryon_train.c native/pryon_qat.c $(SUPPORT)/json.c $(SUPPORT)/hash.c -lm -o $@

# No vendor assets needed to cross-build the trainer. Set ECHO_SOURCE to your echo-dot-assist checkout with NDK r21e.
cross:
	@test -n "$(ECHO_SOURCE)" || { echo 'Set ECHO_SOURCE to the local Echo checkout'; exit 1; }
	@mkdir -p build
	"$(ECHO_SOURCE)/toolchain/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi24-clang" $(CFLAGS) -O3 -mthumb -mfpu=neon -fPIE -pie -I$(SUPPORT) native/pryon_train.c native/pryon_qat.c $(SUPPORT)/json.c $(SUPPORT)/hash.c -lm -o build/pryon-train-arm
