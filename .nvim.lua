local function save_and_run()
  vim.cmd([[w]])
  vim.cmd([[belowright split]])
  vim.cmd([[resize -4]])
  vim.cmd([[terminal cmake -B build && cmake --build build -j16 && ./build/codewars_cpp]])
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
