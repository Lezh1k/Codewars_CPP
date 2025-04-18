local function save_and_run()
  vim.cmd([[w]])
  vim.cmd([[belowright split]])
  vim.cmd([[resize -4]])
  vim.cmd([[terminal cmake -B build && cmake --build build -j16 && ./build/CodewarsCPP]])
end

local function save_and_debug()
  vim.cmd([[wa]])
  vim.cmd([[terminal cmake -S . -B ./build && cmake --build build && gdb -q ./build/CodewarsCPP]])
end

local function save_and_run_unit_tests()
  vim.cmd([[w]])
  vim.cmd([[belowright split]])
  vim.cmd([[resize -4]])
  vim.cmd([[terminal cmake -B build && cmake --build build -j16 && ./build/unit_tests]])
end

local opts = { noremap = true, silent = true }
vim.keymap.set("n", "<C-R>", save_and_run, opts)
vim.keymap.set("n", "<C-T>", save_and_run_unit_tests, opts)
vim.keymap.set("n", "<F5>", save_and_debug, opts)
