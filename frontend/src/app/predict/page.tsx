"use client";

import { useState } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  Home,
  MapPin,
  Square,
  Bed,
  Bath,
  Layout,
  ChevronRight,
  Info,
  Loader2,
  Calculator,
  Plus,
  Minus,
  CheckCircle2,
} from "lucide-react";
import Navbar from "@/components/Navbar";

export default function PredictorPage() {
  const [formData, setFormData] = useState({
    area: "1200",
    bhk: 2,
    bathrooms: 2,
    balcony: 1,
    location: "Electronic City, Bangalore",
  });

  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const locations = [
    "Electronic City, Bangalore",
    "Whitefield, Bangalore",
    "Sarjapur Road, Bangalore",
    "HSR Layout, Bangalore",
    "Koramangala, Bangalore",
    "Indiranagar, Bangalore",
    "Jayanagar, Bangalore",
    "Hebbal, Bangalore",
    "Powai, Mumbai",
    "Andheri West, Mumbai",
    "Bandra West, Mumbai",
    "Juhu, Mumbai",
    "Worli, Mumbai",
    "South Mumbai",
    "Gachibowli, Hyderabad",
    "Kondapur, Hyderabad",
    "Madhapur, Hyderabad",
    "Banjara Hills, Hyderabad",
    "Jubilee Hills, Hyderabad",
    "Gurgaon Sector 56",
    "Gurgaon Sector 45",
    "DLF Phase 3, Gurgaon",
    "Noida Sector 62",
    "Greater Noida West",
    "Salt Lake, Kolkata",
    "New Town, Kolkata",
    "Anna Nagar, Chennai",
    "Adyar, Chennai",
  ];

  const handlePredict = async () => {
    if (parseFloat(formData.area) <= 0) {
      setError("Area must be greater than 0");
      return;
    }
    setError("");
    setLoading(true);
    setPrediction(null);

    try {
      const response = await axios.post("http://localhost:8000/predict", formData);
      setPrediction(response.data);
    } catch (err) {
      setError("Failed to get prediction. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 pt-24 pb-12">
      <Navbar />
      <div className="container mx-auto px-6 lg:px-12 max-w-7xl">
        <header className="mb-12 text-left">
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-3xl font-extrabold text-slate-900 flex items-center gap-3"
          >
            <div className="p-2 bg-blue-100 rounded-xl">
              <Calculator className="text-blue-600 w-6 h-6" />
            </div>
            House Price Predictor
          </motion.h1>
          <p className="text-slate-500 mt-2 ml-14">
            Enter your property details below to get an AI-powered market valuation.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
          {/* Left Column: Form */}
          <div className="lg:col-span-7 space-y-8">
            <section className="bg-white p-8 rounded-[2rem] shadow-sm border border-slate-100">
              <h2 className="text-xl font-bold text-slate-800 mb-8 flex items-center gap-2">
                <Layout className="w-5 h-5 text-blue-500" />
                Property Configuration
              </h2>

              <div className="space-y-10">
                {/* Area Input */}
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
                      <Square className="w-4 h-4 text-blue-500" /> Area (Sq. Ft.)
                    </label>
                    <div className="px-4 py-1.5 bg-blue-50 rounded-full border border-blue-100">
                      <span className="text-blue-700 font-bold text-lg">{formData.area}</span>
                      <span className="text-blue-400 text-xs ml-1">sqft</span>
                    </div>
                  </div>
                  <input
                    type="range"
                    min="200"
                    max="10000"
                    step="50"
                    value={formData.area}
                    onChange={(e) => setFormData({ ...formData, area: e.target.value })}
                    className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase tracking-tighter">
                    <span>200 sqft</span>
                    <span>5,000 sqft</span>
                    <span>10,000 sqft</span>
                  </div>
                </div>

                {/* BHK Selection */}
                <div className="space-y-4">
                  <label className="text-sm font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
                    <Bed className="w-4 h-4 text-blue-500" /> Number of Bedrooms (BHK)
                  </label>
                  <div className="flex flex-wrap gap-3">
                    {[1, 2, 3, 4, 5].map((num) => (
                      <button
                        key={num}
                        onClick={() => setFormData({ ...formData, bhk: num })}
                        className={`px-6 py-3 rounded-2xl font-bold transition-all border-2 ${
                          formData.bhk === num
                            ? "bg-blue-600 text-white border-blue-600 shadow-lg shadow-blue-100 scale-105"
                            : "bg-white text-slate-500 border-slate-100 hover:border-blue-200 hover:text-blue-500"
                        }`}
                      >
                        {num === 5 ? "5+" : num} BHK
                      </button>
                    ))}
                  </div>
                </div>

                {/* Steppers */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
                  <Stepper
                    label="Bathrooms"
                    icon={<Bath className="w-4 h-4 text-blue-500" />}
                    value={formData.bathrooms}
                    onIncrease={() => setFormData({ ...formData, bathrooms: Math.min(10, formData.bathrooms + 1) })}
                    onDecrease={() => setFormData({ ...formData, bathrooms: Math.max(1, formData.bathrooms - 1) })}
                  />
                  <Stepper
                    label="Balcony"
                    icon={<Layout className="w-4 h-4 text-blue-500" />}
                    value={formData.balcony}
                    onIncrease={() => setFormData({ ...formData, balcony: Math.min(5, formData.balcony + 1) })}
                    onDecrease={() => setFormData({ ...formData, balcony: Math.max(0, formData.balcony - 1) })}
                  />
                </div>

                {/* Location Selection */}
                <div className="space-y-4">
                  <label className="text-sm font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-blue-500" /> Location / Area
                  </label>
                  <div className="relative group">
                    <select
                      value={formData.location}
                      onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                      className="w-full appearance-none bg-slate-50 border-2 border-slate-50 rounded-2xl p-4 pr-12 text-slate-700 font-medium focus:bg-white focus:border-blue-500 focus:outline-none transition-all cursor-pointer"
                    >
                      {locations.map((loc) => (
                        <option key={loc} value={loc}>
                          {loc}
                        </option>
                      ))}
                    </select>
                    <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400 group-focus-within:text-blue-500 transition-colors">
                      <ChevronRight className="w-5 h-5 rotate-90" />
                    </div>
                  </div>
                </div>

                <button
                  onClick={handlePredict}
                  disabled={loading}
                  className="w-full group relative overflow-hidden bg-blue-600 text-white py-5 rounded-2xl font-bold text-lg hover:bg-blue-700 transition-all disabled:opacity-70 disabled:cursor-not-allowed shadow-xl shadow-blue-100 flex items-center justify-center gap-3"
                >
                  <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                  {loading ? <Loader2 className="animate-spin w-6 h-6" /> : <Calculator className="w-6 h-6" />}
                  <span>{loading ? "Calculating Value..." : "Reveal Market Prediction"}</span>
                </button>

                {error && (
                  <motion.p
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-red-500 text-sm font-medium bg-red-50 p-4 rounded-xl border border-red-100 text-center"
                  >
                    {error}
                  </motion.p>
                )}
              </div>
            </section>
          </div>

          {/* Right Column: Result */}
          <div className="lg:col-span-5 relative">
            <div className="sticky top-24">
              <AnimatePresence mode="wait">
                {prediction ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, scale: 0.95, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 20 }}
                    className="bg-white p-10 rounded-[2.5rem] shadow-2xl shadow-blue-100/50 border border-blue-50 text-center relative overflow-hidden group"
                  >
                    <div className="absolute -top-10 -right-10 w-40 h-40 bg-blue-50 rounded-full opacity-20 group-hover:scale-150 transition-transform duration-1000" />
                    <div className="absolute -bottom-10 -left-10 w-40 h-40 bg-indigo-50 rounded-full opacity-20 group-hover:scale-150 transition-transform duration-1000" />

                    <div className="relative z-10">
                      <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-50 text-green-600 rounded-full text-xs font-black uppercase tracking-widest mb-6 border border-green-100">
                        <CheckCircle2 className="w-4 h-4" />
                        Analysis Complete
                      </div>

                      <h3 className="text-slate-400 text-sm font-bold uppercase tracking-[0.2em] mb-4">
                        Estimated Market Value
                      </h3>
                      <div className="text-5xl sm:text-6xl font-black text-slate-900 mb-2 tracking-tighter">
                        <span className="text-blue-600">{prediction.formatted_prediction}</span>
                      </div>
                      <p className="text-slate-400 text-sm font-medium mb-10">Calculated for {formData.location}</p>

                      <div className="space-y-6 text-left">
                        <div className="bg-slate-50 p-6 rounded-3xl border border-slate-100">
                          <div className="flex items-start gap-4">
                            <div className="p-2 bg-blue-100 rounded-lg shrink-0">
                              <Info size={20} className="text-blue-600" />
                            </div>
                            <div>
                              <p className="text-sm font-bold text-slate-900 mb-1">Market Insights</p>
                              <p className="text-xs text-slate-500 leading-relaxed italic">
                                "{prediction.influence_summary}"
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="space-y-4">
                          <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
                            Primary Valuation Drivers
                          </h4>
                          <DriverRow label="Location Dynamics" percentage={45} />
                          <DriverRow label="Square Footage" percentage={35} />
                          <DriverRow label="BHK Configuration" percentage={20} />
                        </div>
                      </div>

                      <div className="mt-12 pt-8 border-t border-slate-50 flex flex-col items-center gap-4">
                        <p className="text-[10px] font-bold text-slate-300 uppercase leading-relaxed max-w-[250px]">
                          Powered by our proprietary AI heuristic model for Indian real estate markets.
                        </p>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="h-full min-h-[500px] flex flex-col items-center justify-center p-12 bg-white rounded-[2.5rem] border-2 border-dashed border-slate-100 text-center"
                  >
                    <div className="w-24 h-24 bg-slate-50 rounded-full flex items-center justify-center mb-6">
                      <Home size={40} className="text-slate-200" />
                    </div>
                    <h3 className="text-xl font-bold text-slate-400 mb-2">Ready for Prediction</h3>
                    <p className="text-slate-300 max-w-xs leading-relaxed text-sm font-medium">
                      Fill in the property configuration details and click predict to see the estimated market value.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        <footer className="mt-20 pt-10 border-t border-slate-200 text-center">
          <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">
            &copy; {new Date().getFullYear()} PropPredict Professional &bull; Designed for Indian Markets
          </p>
        </footer>
      </div>
    </div>
  );
}

function Stepper({
  label,
  icon,
  value,
  onIncrease,
  onDecrease,
}: {
  label: string;
  icon: React.ReactNode;
  value: number;
  onIncrease: () => void;
  onDecrease: () => void;
}) {
  return (
    <div className="space-y-4">
      <label className="text-sm font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
        {icon} {label}
      </label>
      <div className="flex items-center justify-between bg-slate-50 rounded-2xl p-2 border-2 border-slate-50 group-focus-within:border-blue-500 transition-all">
        <button
          onClick={onDecrease}
          className="w-12 h-12 rounded-xl bg-white border border-slate-100 flex items-center justify-center text-slate-400 hover:text-blue-600 hover:border-blue-200 hover:shadow-sm active:scale-95 transition-all"
        >
          <Minus className="w-5 h-5" />
        </button>
        <span className="text-2xl font-black text-slate-800 tabular-nums">{value}</span>
        <button
          onClick={onIncrease}
          className="w-12 h-12 rounded-xl bg-white border border-slate-100 flex items-center justify-center text-slate-400 hover:text-blue-600 hover:border-blue-200 hover:shadow-sm active:scale-95 transition-all"
        >
          <Plus className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}

function DriverRow({ label, percentage }: { label: string; percentage: number }) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-end">
        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{label}</span>
        <span className="text-xs font-black text-blue-600">{percentage}%</span>
      </div>
      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: "easeOut", delay: 0.5 }}
          className="h-full bg-gradient-to-r from-blue-500 to-blue-600"
        />
      </div>
    </div>
  );
}
