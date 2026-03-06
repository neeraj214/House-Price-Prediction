"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Calculator } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 bg-white/80 backdrop-blur-md z-50 border-b border-slate-100 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="p-2 bg-blue-500 rounded-lg group-hover:bg-blue-600 transition-colors shadow-blue-200 shadow-md">
              <Home className="text-white w-5 h-5" />
            </div>
            <span className="font-bold text-xl tracking-tight text-slate-800 hidden sm:block">
              Prop<span className="text-blue-600">Predict</span>
            </span>
          </Link>
          <div className="flex items-center gap-2 sm:gap-6">
            <Link
              href="/"
              className={`text-sm font-medium transition-colors hover:text-blue-600 ${
                pathname === "/" ? "text-blue-600" : "text-slate-600"
              }`}
            >
              Home
            </Link>
            <Link
              href="/predict"
              className={`flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-semibold transition-all ${
                pathname === "/predict"
                  ? "bg-blue-600 text-white shadow-lg shadow-blue-200"
                  : "bg-blue-50 text-blue-600 hover:bg-blue-100"
              }`}
            >
              <Calculator className="w-4 h-4" />
              <span>Predict Now</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
